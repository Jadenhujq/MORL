# esg_env.py
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class ESGPortfolioEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        returns,
        esg_scores,
        features,
        transaction_cost=1e-3
    ):
        self.returns = returns
        self.esg_scores = esg_scores
        self.features = features
        self.n_steps, self.n_assets_plus = returns.shape
        self.n_assets = self.n_assets_plus - 1
        self.transaction_cost = transaction_cost

        obs_dim = (
            1 +
            self.n_assets_plus +
            features["rolling_mean"].shape[1] +
            features["rolling_vol"].shape[1] +
            features["factors"].shape[1] +
            features["esg_factors"].shape[1] +
            self.n_assets_plus
        )

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.n_assets_plus,), dtype=np.float32
        )

    def _get_obs(self):
        return np.concatenate([
            [np.log(self.portfolio_value)],
            self.current_weights,
            self.features["rolling_mean"][self.t],
            self.features["rolling_vol"][self.t],
            self.features["factors"][self.t],
            self.features["esg_factors"][self.t],
            self.esg_scores[self.t],
        ]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        self.portfolio_value = 1.0
        self.current_weights = np.zeros(self.n_assets_plus)
        self.current_weights[0] = 1.0
        return self._get_obs(), {}

    def step(self, action, weights_obj=(0.6, 0.3, 0.1)):
        alpha, beta, lam = weights_obj
        action = np.clip(action, 1e-8, np.inf)
        target_weights = action / action.sum()

        gross_returns = self.returns[self.t]
        esg = self.esg_scores[self.t]

        turnover = np.abs(target_weights - self.current_weights).sum()
        cost = self.transaction_cost * turnover

        next_value = self.portfolio_value * np.dot(target_weights, gross_returns) * (1 - cost)
        log_return = np.log(next_value / self.portfolio_value)

        if self.t > 1:
            cov_est = np.cov(
                np.log(self.returns[max(0, self.t-59):self.t+1, 1:]).T
            )
        else:
            cov_est = np.eye(self.n_assets)
        risk_term = - float(self.current_weights[1:].T @ cov_est @ self.current_weights[1:])
        esg_term = float(self.current_weights @ esg)

        reward = alpha * log_return + beta * risk_term + lam * esg_term

        self.portfolio_value = next_value
        self.current_weights = target_weights
        self.t += 1
        terminated = self.t >= self.n_steps - 1

        return self._get_obs(), reward, terminated, False, {
            "log_return": log_return,
            "risk_term": risk_term,
            "esg_term": esg_term,
            "portfolio_value": self.portfolio_value,
            "turnover": turnover,
        }
