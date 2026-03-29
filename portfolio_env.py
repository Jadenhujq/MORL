import numpy as np
import gymnasium as gym
from gymnasium import spaces


class PortfolioEnv(gym.Env):
    """
    Multi-objective Portfolio Optimization Environment (MORL-compatible)

    Reward vector:
        r1: return
        r2: risk (negative variance)
        r3: ESG score

    Scalar reward:
        u = w · r
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        returns,
        cov_matrices,
        esg_scores,
        weight_pref=(0.33, 0.33, 0.33),
        transaction_cost=0.001,
    ):
        super().__init__()

        # ===== 数据 =====
        self.returns = returns                  # shape: (T, N)
        self.cov_matrices = cov_matrices        # shape: (T, N, N)
        self.esg_scores = esg_scores            # shape: (N,)

        self.T, self.n_assets = returns.shape

        # ===== MORL 偏好 =====
        self.weight_pref = np.array(weight_pref)

        # ===== 交易成本 =====
        self.transaction_cost = transaction_cost

        # ===== 状态 =====
        self.t = 0
        self.weights = None

        # ===== Action space =====
        # continuous portfolio weights
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.n_assets,),
            dtype=np.float32
        )

        # ===== Observation space =====
        # 用 returns + 当前 weights
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.n_assets * 2,),
            dtype=np.float32
        )

    # =========================================================
    # Reset
    # =========================================================
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.t = 0

        # 初始均匀分配
        self.weights = np.ones(self.n_assets) / self.n_assets

        return self._get_obs(), {}

    # =========================================================
    # Step
    # =========================================================
    def step(self, action):

        # ===== 1. 处理 action（防 NaN）=====
        action = np.nan_to_num(action, nan=1e-6)
        action = np.clip(action, 1e-6, 1.0)

        weights = action / np.sum(action)

        # ===== 2. 当前市场 =====
        ret = self.returns[self.t]
        cov = self.cov_matrices[self.t]

        # ===== 3. Portfolio return =====
        portfolio_return = float(weights @ ret)

        # 防止 log 崩溃
        portfolio_return = np.clip(portfolio_return, -0.99, None)

        # ===== 4. Reward components =====
        r1 = np.log(1 + portfolio_return)

        r2 = -float(weights.T @ cov @ weights)

        r3 = float(weights @ self.esg_scores)

        reward_vector = np.array([r1, r2, r3], dtype=np.float32)

        # ===== 5. Transaction cost =====
        cost = self.transaction_cost * np.sum(np.abs(weights - self.weights))

        # ===== 6. Scalar reward（给 PPO）=====
        scalar_reward = float(self.weight_pref @ reward_vector - cost)

        # ===== 7. 更新状态 =====
        self.weights = weights
        self.t += 1

        done = self.t >= self.T - 1

        # ===== 8. Next state =====
        obs = self._get_obs()

        # ===== 9. Info（用于分析）=====
        info = {
            "reward_vector": reward_vector,
            "return": r1,
            "risk": r2,
            "esg": r3,
            "cost": cost,
        }

        return obs, scalar_reward, done, False, info

    # =========================================================
    # Observation
    # =========================================================
    def _get_obs(self):
        return np.concatenate([
            self.returns[self.t],
            self.weights
        ]).astype(np.float32)

