import importlib.util
import os
import warnings
from pathlib import Path

import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3

matplotlib.use("Agg")
warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
RUN_TAG = os.getenv("RUN_TAG", "").strip()
RUN_SUFFIX = f"_{RUN_TAG}" if RUN_TAG else ""
MODEL_DIR = BASE_DIR / f"trained_models{RUN_SUFFIX}"
DEBUG_TD3 = os.getenv("DEBUG_TD3", "0").strip() == "1"

INDICATORS = [
    "macd",
    "boll_ub",
    "boll_lb",
    "rsi_30",
    "cci_30",
    "dx_30",
    "close_30_sma",
    "close_60_sma",
]


def load_stock_trading_env():
    env_path = BASE_DIR.parent / "FinRL/finrl/meta/env_stock_trading/env_stocktrading.py"
    spec = importlib.util.spec_from_file_location("finrl_env_stocktrading", env_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.StockTradingEnv


StockTradingEnv = load_stock_trading_env()


class FixedStockEnv(StockTradingEnv):
    def __init__(self, df, **kwargs):
        df = df.copy()
        df["close"] = df["close"].astype(float).replace(0, 0.01).fillna(0.01)
        super().__init__(df=df, **kwargs)
        self.stock_dim = kwargs.get("stock_dim")

    def _initiate_state(self):
        start_idx = self.day * self.stock_dim
        end_idx = (self.day + 1) * self.stock_dim
        self.data = self.df.iloc[start_idx:end_idx, :]
        state = [self.initial_amount]
        state += self.data.close.values.tolist()
        state += [0] * self.stock_dim
        state += sum([self.data[tech].values.tolist() for tech in self.tech_indicator_list], [])
        return [0.01 if x == 0 or pd.isna(x) else float(x) for x in state]

    def _update_state(self):
        start_idx = self.day * self.stock_dim
        end_idx = (self.day + 1) * self.stock_dim
        self.data = self.df.iloc[start_idx:end_idx, :]
        state = [self.state[0]]
        state += self.data.close.values.tolist()
        state += list(self.state[(self.stock_dim + 1) : (2 * self.stock_dim + 1)])
        state += sum([self.data[tech].values.tolist() for tech in self.tech_indicator_list], [])
        return state

    def _buy_stock(self, index, action):
        available_amount = self.state[0] // (self.state[index + 1] * (1 + self.buy_cost_pct[index]))
        quantity = min(available_amount, action)
        self.state[0] -= self.state[index + 1] * quantity * (1 + self.buy_cost_pct[index])
        self.state[index + self.stock_dim + 1] += quantity
        return quantity

    def _sell_stock(self, index, action):
        quantity = min(abs(action), self.state[index + self.stock_dim + 1])
        self.state[0] += self.state[index + 1] * quantity * (1 - self.sell_cost_pct[index])
        self.state[index + self.stock_dim + 1] -= quantity
        return quantity


class ConditionedMORLWrapper(gym.Wrapper):
    def __init__(self, env, weight_dim=3, target_base_shape=51):
        super().__init__(env)
        self.weight_dim = weight_dim
        self.target_base_shape = target_base_shape
        self.current_weights = np.array([0.33, 0.33, 0.34], dtype=np.float32)

    def _normalize_obs(self, obs):
        obs = np.array(obs, dtype=np.float32)
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        obs = np.clip(obs / 1e5, -5, 5)
        if obs.shape[0] < self.target_base_shape:
            obs = np.concatenate([obs, np.zeros(self.target_base_shape - obs.shape[0], dtype=np.float32)])
        elif obs.shape[0] > self.target_base_shape:
            obs = obs[: self.target_base_shape]
        return obs.astype(np.float32)

    def reset(self, **kwargs):
        res = self.env.reset(**kwargs)
        obs = res[0] if isinstance(res, tuple) else res
        normalized_obs = self._normalize_obs(obs)
        return np.concatenate([normalized_obs, self.current_weights]).astype(np.float32)

    def step(self, action):
        res = self.env.step(action)
        if len(res) == 5:
            obs, reward, terminated, truncated, info = res
        else:
            obs, reward, done, info = res
            terminated, truncated = done, False
        normalized_obs = self._normalize_obs(obs)
        return (
            np.concatenate([normalized_obs, self.current_weights]).astype(np.float32),
            reward,
            terminated,
            truncated,
            info,
        )
def run_conditioned_backtest(model, trade_df, env_kwargs, weights):
    env_trade_raw = FixedStockEnv(df=trade_df, **env_kwargs)
    env_trade = ConditionedMORLWrapper(
        env_trade_raw,
        weight_dim=3,
        target_base_shape=env_kwargs["state_space"],
    )
    env_trade.current_weights = np.array(weights, dtype=np.float32)

    obs = env_trade.reset()
    account_values = [env_kwargs["initial_amount"]]
    hsiesg_allocations = [0.0]
    portfolio_esg_scores = [0.0]
    last_action = np.zeros(env_kwargs["stock_dim"])
    unique_dates = trade_df.date.unique()

    for _ in range(len(unique_dates) - 1):
        try:
            action, _ = model.predict(obs, deterministic=True)
            action = (0.7 * action) + (0.3 * last_action)
            last_action = action

            current_cash = env_trade.env.state[0]
            if current_cash < (env_kwargs["initial_amount"] * 0.05):
                action = np.clip(action, -1, 0)

            obs, _, terminated, truncated, _ = env_trade.step(action)
            current_val = env_trade.env.asset_memory[-1]
            if abs(current_val - account_values[-1]) > (account_values[-1] * 0.15):
                current_val = account_values[-1]
            account_values.append(current_val)

            shares = np.array(
                env_trade.env.state[env_trade.env.stock_dim + 1 : 2 * env_trade.env.stock_dim + 1],
                dtype=float,
            )
            prices = np.array(env_trade.env.state[1 : env_trade.env.stock_dim + 1], dtype=float)
            cash = float(env_trade.env.state[0])
            stock_values = prices * shares
            total_value = max(cash + float(stock_values.sum()), 1e-8)
            asset_weights = stock_values / total_value

            tic_list = env_trade.env.data["tic"].tolist()
            total_shares = shares.sum()
            if total_shares > 0 and "HSIESG" in tic_list:
                idx = tic_list.index("HSIESG")
                hsiesg_allocations.append(float(shares[idx] / total_shares))
            else:
                hsiesg_allocations.append(0.0)

            if "esg_score" in env_trade.env.data.columns:
                esg_scores = env_trade.env.data["esg_score"].to_numpy(dtype=float)
                portfolio_esg_scores.append(float(asset_weights @ esg_scores))
            else:
                portfolio_esg_scores.append(0.0)

            if terminated or truncated:
                break
        except Exception:
            account_values.append(account_values[-1])
            hsiesg_allocations.append(hsiesg_allocations[-1])
            portfolio_esg_scores.append(portfolio_esg_scores[-1])

    result = pd.DataFrame(
        {
            "date": unique_dates[: len(account_values)],
            "account_value": account_values,
            "hsiesg_share": hsiesg_allocations[: len(account_values)],
            "portfolio_esg_score": portfolio_esg_scores[: len(account_values)],
        }
    ).set_index("date")
    return result


def debug_td3_preferences(model, trade_df, env_kwargs, preference_profiles, num_days=10):
    print("\n=== TD3 Preference Debug (first 10 steps) ===")
    for profile_name, weights in preference_profiles.items():
        env_trade_raw = FixedStockEnv(df=trade_df, **env_kwargs)
        env_trade = ConditionedMORLWrapper(
            env_trade_raw,
            weight_dim=3,
            target_base_shape=env_kwargs["state_space"],
        )
        env_trade.current_weights = np.array(weights, dtype=np.float32)
        obs = env_trade.reset()
        last_action = np.zeros(env_kwargs["stock_dim"], dtype=np.float32)
        unique_dates = trade_df.date.unique()

        print(f"\n[{profile_name}]")
        for step_idx in range(min(num_days, max(len(unique_dates) - 1, 0))):
            action, _ = model.predict(obs, deterministic=True)
            smoothed_action = (0.7 * action) + (0.3 * last_action)
            last_action = smoothed_action

            obs_tail = np.round(obs[-3:], 4).tolist()
            action_head = np.round(action, 4).tolist()
            smoothed_head = np.round(smoothed_action, 4).tolist()
            print(
                f"day={step_idx + 1:02d} date={unique_dates[step_idx]} "
                f"obs_pref={obs_tail} raw_action={action_head} smooth_action={smoothed_head}"
            )

            current_cash = env_trade.env.state[0]
            exec_action = smoothed_action
            if current_cash < (env_kwargs["initial_amount"] * 0.05):
                exec_action = np.clip(exec_action, -1, 0)

            obs, _, terminated, truncated, _ = env_trade.step(exec_action)
            if terminated or truncated:
                break


def summarize_series(name, profile_name, series, esg_series, portfolio_esg_series):
    ret = series.pct_change().fillna(0)
    total_return = series.iloc[-1] / series.iloc[0] - 1
    max_drawdown = (series / series.cummax() - 1).min()
    ann_vol = ret.std() * np.sqrt(252)
    ann_ret = (1 + total_return) ** (252 / max(len(series) - 1, 1)) - 1
    sharpe_like = ann_ret / ann_vol if ann_vol > 0 else np.nan
    return {
        "model": name,
        "preference_profile": profile_name,
        "start_value": float(series.iloc[0]),
        "final_value": float(series.iloc[-1]),
        "total_return_pct": float(total_return * 100),
        "max_drawdown_pct": float(max_drawdown * 100),
        "ann_vol_pct": float(ann_vol * 100),
        "ann_return_pct": float(ann_ret * 100),
        "sharpe_like": float(sharpe_like) if np.isfinite(sharpe_like) else np.nan,
        "avg_hsiesg_share": float(esg_series.mean() * 100),
        "avg_portfolio_esg_score": float(portfolio_esg_series.mean()),
    }


def main() -> None:
    trade = pd.read_csv(BASE_DIR / "trade_data.csv")
    trade["date"] = trade["date"].astype(str)
    trade = trade.sort_values(["date", "tic"]).reset_index(drop=True)

    stock_dimension = trade.tic.nunique()
    expected_base_state = 1 + 2 * stock_dimension + (len(INDICATORS) * stock_dimension)
    env_kwargs = {
        "hmax": 100,
        "initial_amount": 1_000_000,
        "num_stock_shares": [0] * stock_dimension,
        "buy_cost_pct": [0.001] * stock_dimension,
        "sell_cost_pct": [0.001] * stock_dimension,
        "state_space": expected_base_state,
        "stock_dim": stock_dimension,
        "tech_indicator_list": INDICATORS,
        "action_space": stock_dimension,
        "reward_scaling": 1e-6,
    }

    preference_profiles = {
        "Balanced": [0.33, 0.33, 0.34],
        "ReturnTilt": [0.70, 0.20, 0.10],
        "RiskTilt": [0.20, 0.70, 0.10],
        "ESGTilt": [0.20, 0.20, 0.60],
    }

    agents = {"PPO": PPO, "A2C": A2C, "DDPG": DDPG, "TD3": TD3, "SAC": SAC}
    loaded_models = {}
    for model_name, model_cls in agents.items():
        path = MODEL_DIR / f"agent_{model_name.lower()}_morl.zip"
        if path.exists():
            print(f"Loading {model_name} from {path}")
            loaded_models[model_name] = model_cls.load(path)

    if DEBUG_TD3 and "TD3" in loaded_models:
        debug_td3_preferences(loaded_models["TD3"], trade, env_kwargs, preference_profiles, num_days=10)

    results_dict = {}
    summary_rows = []
    for profile_name, weights in preference_profiles.items():
        for model_name, model in loaded_models.items():
            result = run_conditioned_backtest(model, trade, env_kwargs, weights)
            results_dict[f"{model_name} | {profile_name}"] = result["account_value"]
            summary_rows.append(
                summarize_series(
                    model_name,
                    profile_name,
                    result["account_value"],
                    result["hsiesg_share"],
                    result["portfolio_esg_score"],
                )
            )

    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(
            ["preference_profile", "total_return_pct", "sharpe_like"],
            ascending=[True, False, False],
        ).reset_index(drop=True)
        summary_path = BASE_DIR / f"active_esg_backtest_summary{RUN_SUFFIX}.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"Saved {summary_path}")

    if results_dict:
        n_profiles = len(preference_profiles)
        ncols = 2
        nrows = (n_profiles + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(16, 5 * nrows), sharex=True, sharey=True)
        axes = np.array(axes).reshape(-1)
        for idx, (profile_name, _) in enumerate(preference_profiles.items()):
            ax = axes[idx]
            for model_name in loaded_models.keys():
                key = f"{model_name} | {profile_name}"
                if key in results_dict:
                    ax.plot(results_dict[key].index, results_dict[key].values, label=model_name, linewidth=2)
            ax.set_title(profile_name)
            ax.set_ylabel("Portfolio Value (HKD)")
            ax.tick_params(axis="x", rotation=45)
            ax.grid(True, alpha=0.3)
            ax.xaxis.set_major_locator(plt.MaxNLocator(8))
            ax.legend()

        for idx in range(n_profiles, len(axes)):
            axes[idx].axis("off")

        fig.suptitle("Preference-Conditioned Active ESG Backtest Comparison", fontsize=16)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        plot_path = BASE_DIR / f"active_esg_5_asset_backtest{RUN_SUFFIX}.png"
        plt.savefig(plot_path, dpi=300)
        print(f"Saved {plot_path}")


if __name__ == "__main__":
    main()
