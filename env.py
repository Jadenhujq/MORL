import numpy as np
import gymnasium as gym
from gymnasium import Env
from gymnasium.spaces import Box
from gymnasium.utils import EzPickle


class PortfolioEnv(Env, EzPickle):

    metadata = {"render_modes": []}

    def __init__(
        self,
        returns_data,        # (T, N) 历史收益率数据
        esg_scores,          # (N,) 每个资产的 ESG 分数
        window_size=20,      # 用多少历史数据作为 state
        risk_aversion=0.1,   # 风险权重（目前未直接用，可扩展）
        transaction_cost=0.001  # 每次调仓成本
    ):

        EzPickle.__init__(self)

        # ===== 数据 =====
        self.returns_data = returns_data
        self.esg_scores = esg_scores

        # ===== 基本维度 =====
        self.num_assets = returns_data.shape[1]   # N（资产数量）
        self.num_steps = returns_data.shape[0]    # T（时间长度）

        # ===== 参数 =====
        self.window_size = window_size
        self.risk_aversion = risk_aversion
        self.transaction_cost = transaction_cost

        # 当前时间步（从window开始）
        self.current_step = window_size

        # ===== Observation Space =====
        # state = [过去window的returns + 当前权重]
        self.observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(window_size * self.num_assets + self.num_assets,),
            dtype=np.float32
        )

        # ===== Action Space =====
        # action = portfolio weights（0~1）
        self.action_space = Box(
            low=0,
            high=1,
            shape=(self.num_assets,),
            dtype=np.float32
        )

        # ===== 多目标 reward =====
        self.reward_dim = 3
        self.num_objectives = 3

        # 初始权重（等权）
        self.weights = np.ones(self.num_assets) / self.num_assets

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        # 重置时间
        self.current_step = self.window_size

        # 重置为等权组合
        self.weights = np.ones(self.num_assets) / self.num_assets

        # 返回初始 state
        return self._get_obs(), {}

    def _get_obs(self):

        # 取过去 window_size 的收益率
        window_returns = self.returns_data[
            self.current_step - self.window_size:self.current_step
        ].flatten()   # 展平成一维

        # 拼接当前权重
        obs = np.concatenate([window_returns, self.weights])

        return obs.astype(np.float32)

    def step(self, action):

        # ===== 1. 处理 action =====
        action = np.clip(action, 0, 1)
        action = action + 1e-8
        action = action / np.sum(action)

        prev_weights = self.weights
        self.weights = action

        # ===== 2. 收益 =====
        returns = self.returns_data[self.current_step]
        portfolio_return = np.dot(self.weights, returns)

        # ===== 3. 风险 =====
        window_data = self.returns_data[
            self.current_step - self.window_size:self.current_step
        ]

        cov = np.cov(window_data.T)
        risk = self.weights @ cov @ self.weights

        # ===== 4. ESG =====
        esg_score = np.dot(self.weights, self.esg_scores)

        # ===== 5. 交易成本 =====
        turnover = np.sum(np.abs(self.weights - prev_weights))
        cost = turnover * self.transaction_cost
        portfolio_return -= cost

        # ===== 6. reward vector =====
        reward = np.array([
            portfolio_return * 50,
            -risk * 1,
            esg_score * 5
        ])

        # ===== 7. info（给 evaluate 用）=====
        info = {
            "return": float(portfolio_return),
            "risk": float(risk),
            "esg": float(esg_score)
        }

        # ===== 8. 时间推进 =====
        self.current_step += 1
        terminated = self.current_step >= self.num_steps - 1

        # ===== 9. 正确的 observation =====
        obs = self._get_obs()

        return obs, reward, terminated, False, info

    def render(self):
        pass