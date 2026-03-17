import numpy as np
import gymnasium as gym
from gymnasium import Env
from gymnasium.spaces import Box
from gymnasium.utils import EzPickle


class PortfolioEnv(Env, EzPickle):

    metadata = {"render_modes": []}

    def __init__(
        self,
        returns_data,
        esg_scores,
        window_size=20,
        risk_aversion=0.1,
        transaction_cost=0.001
    ):

        EzPickle.__init__(self)

        self.returns_data = returns_data
        self.esg_scores = esg_scores

        self.num_assets = returns_data.shape[1]
        self.num_steps = returns_data.shape[0]

        self.window_size = window_size
        self.risk_aversion = risk_aversion
        self.transaction_cost = transaction_cost

        self.current_step = window_size

        # observation space
        self.observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(window_size * self.num_assets + self.num_assets,),
            dtype=np.float32
        )

        # action = portfolio weights
        self.action_space = Box(
            low=0,
            high=1,
            shape=(self.num_assets,),
            dtype=np.float32
        )

        # reward dimension
        self.reward_dim = 3
        self.num_objectives = 3

        self.weights = np.ones(self.num_assets) / self.num_assets

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        self.current_step = self.window_size
        self.weights = np.ones(self.num_assets) / self.num_assets

        return self._get_obs(), {}

    def _get_obs(self):

        window_returns = self.returns_data[
            self.current_step - self.window_size:self.current_step
        ].flatten()

        obs = np.concatenate([window_returns, self.weights])

        return obs.astype(np.float32)

    def step(self, action):

        action = np.clip(action, 0, 1)

        # normalize weights
        action = action / np.sum(action)

        prev_weights = self.weights
        self.weights = action

        returns = self.returns_data[self.current_step]

        portfolio_return = np.dot(self.weights, returns)

        # risk (variance proxy)
        risk = np.var(
            self.returns_data[
                self.current_step - self.window_size:self.current_step
            ] @ self.weights
        )

        # ESG score
        esg_score = np.dot(self.weights, self.esg_scores)

        # transaction cost
        turnover = np.sum(np.abs(self.weights - prev_weights))
        cost = turnover * self.transaction_cost

        portfolio_return -= cost

        reward = np.array([
            portfolio_return,
            -risk,
            esg_score
        ])

        self.current_step += 1

        terminated = self.current_step >= self.num_steps - 1

        return self._get_obs(), reward, terminated, False, {}

    def render(self):
        pass