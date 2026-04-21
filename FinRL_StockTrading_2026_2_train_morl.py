import importlib.util
import os
import warnings
from pathlib import Path

import gymnasium as gym
import matplotlib
import numpy as np
import pandas as pd
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3
from stable_baselines3.common.noise import NormalActionNoise

matplotlib.use("Agg")
warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
RUN_TAG = os.getenv("RUN_TAG", "").strip()
RUN_SUFFIX = f"_{RUN_TAG}" if RUN_TAG else ""
MODEL_DIR = BASE_DIR / f"trained_models{RUN_SUFFIX}"
RESULTS_DIR = BASE_DIR / f"results{RUN_SUFFIX}"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
GLOBAL_SEED = int(os.getenv("SEED", "42"))
TIMESTEP_MULTIPLIER = float(os.getenv("TIMESTEP_MULTIPLIER", "1.0"))
ACTION_NOISE_SIGMA = float(os.getenv("ACTION_NOISE_SIGMA", "0.05"))
RISK_PENALTY_COEF = float(os.getenv("RISK_PENALTY_COEF", "0.05"))
ESG_REWARD_COEF = float(os.getenv("ESG_REWARD_COEF", "0.08"))

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
        self.state_space = kwargs.get("state_space")

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

    def step(self, actions):
        actions = np.clip(actions, -1, 1)
        actions = np.nan_to_num(actions, nan=0.0)
        try:
            return super().step(actions)
        except Exception:
            return self.state, 0.0, False, False, {}


class ConditionedMORLWrapper(gym.Wrapper):
    def __init__(self, env, weight_dim=3):
        super().__init__(env)
        self.weight_dim = weight_dim
        self.target_base_shape = env.observation_space.shape[0]
        self.observation_space = gym.spaces.Box(
            low=-5.0,
            high=5.0,
            shape=(self.target_base_shape + weight_dim,),
            dtype=np.float32,
        )
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

    def _sample_weights(self):
        anchor_profiles = [
            np.array([0.33, 0.33, 0.34], dtype=np.float32),
            np.array([0.70, 0.20, 0.10], dtype=np.float32),
            np.array([0.20, 0.70, 0.10], dtype=np.float32),
            np.array([0.20, 0.20, 0.60], dtype=np.float32),
        ]
        rand_val = np.random.rand()
        if rand_val < 0.60:
            return anchor_profiles[np.random.randint(len(anchor_profiles))].copy()
        if rand_val < 0.75:
            return np.array([1.0, 0.0, 0.0], dtype=np.float32)
        if rand_val < 0.90:
            return np.array([0.0, 1.0, 0.0], dtype=np.float32)
        return np.random.dirichlet(np.ones(self.weight_dim)).astype(np.float32)

    def _extract_portfolio_vectors(self):
        state = list(self.env.state) if isinstance(self.env.state, (list, tuple, np.ndarray)) else []
        stock_dim = self.env.stock_dim

        prices = np.array(state[1 : 1 + stock_dim], dtype=np.float32) if len(state) >= 1 + stock_dim else np.zeros(stock_dim, dtype=np.float32)
        shares = np.array(
            state[stock_dim + 1 : 2 * stock_dim + 1],
            dtype=np.float32,
        ) if len(state) >= 2 * stock_dim + 1 else np.zeros(stock_dim, dtype=np.float32)
        cash = float(state[0]) if len(state) >= 1 else float(self.env.initial_amount)
        return cash, prices, shares

    def _current_esg_scores(self):
        data = getattr(self.env, "data", None)
        if isinstance(data, pd.DataFrame) and not data.empty and "esg_score" in data.columns:
            scores = data["esg_score"].to_numpy(dtype=np.float32)
            if len(scores) == self.env.stock_dim:
                return scores
        return np.zeros(self.env.stock_dim, dtype=np.float32)

    def reset(self, **kwargs):
        self.current_weights = self._sample_weights()
        res = self.env.reset(**kwargs)
        obs = res[0] if isinstance(res, tuple) else res
        normalized_obs = self._normalize_obs(obs)
        return np.concatenate([normalized_obs, self.current_weights]).astype(np.float32), {}

    def step(self, action):
        res = self.env.step(action)
        if len(res) == 5:
            obs, reward, terminated, truncated, info = res
        else:
            obs, reward, done, info = res
            terminated, truncated = done, False

        normalized_obs = self._normalize_obs(obs)
        account_values = self.env.asset_memory
        prev_value = float(account_values[-2]) if len(account_values) >= 2 else float(self.env.initial_amount)
        current_value = float(account_values[-1]) if len(account_values) >= 1 else prev_value
        prev_value = max(prev_value, 1e-8)
        current_value = max(current_value, 1e-8)
        return_signal = float(reward)

        cash, prices, shares = self._extract_portfolio_vectors()
        risk_penalty = -RISK_PENALTY_COEF * float(np.mean(np.abs(normalized_obs)))

        stock_values = prices * shares
        total_value = max(cash + float(stock_values.sum()), 1e-8)
        asset_weights = stock_values / total_value
        esg_scores = self._current_esg_scores()
        portfolio_esg_score = float(asset_weights @ esg_scores)
        esg_reward_signal = ESG_REWARD_COEF * portfolio_esg_score

        conditioned_reward = (
            self.current_weights[0] * return_signal
            + self.current_weights[1] * risk_penalty
            + self.current_weights[2] * esg_reward_signal
        )
        final_reward = float(np.clip(conditioned_reward, -1.0, 1.0))
        info = dict(info)
        info["reward_components"] = {
            "return_signal": return_signal,
            "risk_penalty": risk_penalty,
            "esg_reward_signal": esg_reward_signal,
            "portfolio_esg_score": portfolio_esg_score,
            "weights": self.current_weights.tolist(),
        }
        return (
            np.concatenate([normalized_obs, self.current_weights]).astype(np.float32),
            final_reward,
            terminated,
            truncated,
            info,
        )


def build_train_dataframe() -> pd.DataFrame:
    train = pd.read_csv(BASE_DIR / "train_data.csv")
    train["date"] = pd.to_datetime(train["date"])
    pivoted = train.pivot(index="date", columns="tic").ffill().bfill()
    train = pivoted.stack(level=1, future_stack=True).reset_index()
    train.columns = [str(c).lower() for c in train.columns]
    train = train.rename(columns={"level_0": "date", "level_1": "tic"})
    train = train.sort_values(["date", "tic"]).reset_index(drop=True)
    numeric_cols = [col for col in train.columns if col not in {"date", "tic"}]
    train[numeric_cols] = train[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0.01)
    return train
def main() -> None:
    np.random.seed(GLOBAL_SEED)
    train = build_train_dataframe()

    stock_dimension = train["tic"].nunique()
    expected_base_state = 1 + 2 * stock_dimension + (stock_dimension * len(INDICATORS))

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
        "turbulence_threshold": None,
    }

    def make_env():
        return ConditionedMORLWrapper(FixedStockEnv(df=train, **env_kwargs), weight_dim=3)

    def train_and_save(name, model_cls, total_timesteps, model_kwargs):
        print(f"--- Training {name.upper()} ---")
        adjusted_timesteps = max(1000, int(round(total_timesteps * TIMESTEP_MULTIPLIER)))
        model = model_cls("MlpPolicy", make_env(), verbose=1, seed=GLOBAL_SEED, **model_kwargs)
        model.learn(total_timesteps=adjusted_timesteps, progress_bar=False)
        model.save(MODEL_DIR / f"agent_{name}_morl")

    training_specs = [
        ("a2c", A2C, 12_000, {"learning_rate": 1e-4, "n_steps": 10}),
        ("ppo", PPO, 12_000, {"learning_rate": 1e-4, "n_steps": 2048}),
        (
            "ddpg",
            DDPG,
            8_000,
            {
                "learning_rate": 5e-5,
                "buffer_size": 20_000,
                "batch_size": 128,
                "learning_starts": 1_000,
                "train_freq": 1,
                "gradient_steps": 1,
                "action_noise": NormalActionNoise(
                    mean=np.zeros(stock_dimension),
                    sigma=ACTION_NOISE_SIGMA * np.ones(stock_dimension),
                ),
            },
        ),
        (
            "td3",
            TD3,
            8_000,
            {
                "learning_rate": 5e-5,
                "buffer_size": 20_000,
                "batch_size": 128,
                "learning_starts": 1_000,
                "train_freq": 1,
                "gradient_steps": 1,
                "action_noise": NormalActionNoise(
                    mean=np.zeros(stock_dimension),
                    sigma=ACTION_NOISE_SIGMA * np.ones(stock_dimension),
                ),
            },
        ),
        (
            "sac",
            SAC,
            6_000,
            {
                "learning_rate": 3e-5,
                "buffer_size": 30_000,
                "batch_size": 128,
                "learning_starts": 1_000,
                "train_freq": 1,
                "gradient_steps": 1,
                "ent_coef": "auto_0.1",
            },
        ),
    ]

    for name, model_cls, total_timesteps, model_kwargs in training_specs:
        train_and_save(name, model_cls, total_timesteps, model_kwargs)

    print(f"Saved models to {MODEL_DIR}")


if __name__ == "__main__":
    main()
