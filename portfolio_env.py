import gymnasium as gym
import numpy as np
from gymnasium import spaces

class DynamicPortfolioEnv(gym.Env):

    def __init__(
        self,
        returns,
        esg_scores,
        cov_matrices,
        features,
        rho=1.0,
        zeta=0.1,
        phi=0.1,
        initial_value=1.0
    ):
        super().__init__()

        self.returns = returns
        self.esg_scores = esg_scores
        self.cov = cov_matrices
        self.features = features

        self.T = len(returns)
        self.n_assets = returns.shape[1]

        self.rho = rho
        self.zeta = zeta
        self.phi = phi

        self.initial_value = initial_value

        self.action_space = spaces.Box(
            low=0,
            high=1,
            shape=(self.n_assets,),
            dtype=np.float32
        )

        state_dim = 1 + self.n_assets + features.shape[1] + self.n_assets

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(state_dim,),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        self.V = self.initial_value
        self.w = np.ones(self.n_assets) / self.n_assets
        state = self._get_state()
        return state, {}

    def step(self, action):
        weights = self._normalize(action)

        Rt = self.returns[self.t]
        cov = self.cov[self.t]
        esg = self.esg_scores[self.t]

        portfolio_return = np.dot(weights, Rt)
        V_next = self.V * portfolio_return

        r1 = np.log(V_next / self.V)
        r2 = - weights.T @ cov @ weights
        r3 = weights @ esg

        reward = self.rho * r1 + self.zeta * r2 + self.phi * r3

        self.V = V_next
        self.w = weights
        self.t += 1

        terminated = self.t >= self.T - 1
        state = self._get_state() if not terminated else np.zeros(self.observation_space.shape)

        info = {
            "vector_reward": np.array([r1, r2, r3])
        }

        return state, reward, terminated, False, info

    def _get_state(self):
        Xt = self.features[self.t]
        et = self.esg_scores[self.t]
        return np.concatenate([
            [self.V],
            self.w,
            Xt,
            et
        ]).astype(np.float32)

    def _normalize(self, action):
        action = np.maximum(action, 1e-8)
        return action / np.sum(action)
    

if __name__ == "__main__":
    import numpy as np

    # 测试用
    T = 50      # 50个时间步
    n = 3       # 3个资产
    d = 5       # 5个特征

    returns = np.random.normal(1.01, 0.02, (T, n))
    esg_scores = np.random.uniform(0, 10, (T, n))
    cov_matrices = np.random.randn(T, n, n) * 0.001
    features = np.random.randn(T, d)

    env = DynamicPortfolioEnv(returns, esg_scores, cov_matrices, features)

    # 开始测试
    obs, _ = env.reset()
    print("初始状态:", obs)

    for i in range(10):
        action = env.action_space.sample()
        obs, reward, done, _, info = env.step(action)

        print(f"步骤 {i+1} | 奖励: {reward:.4f} | 资产价值: {env.V:.3f}")
        if done:
            break