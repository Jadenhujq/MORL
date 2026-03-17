import gymnasium as gym
import numpy as np

class ScalarizationWrapper(gym.Wrapper):
    def __init__(self, env, weights):
        super().__init__(env)
        self.weights = np.array(weights)

    def step(self, action):
        obs, reward_vec, terminated, truncated, info = self.env.step(action)

        # 多目标 → 单目标
        reward = float(np.dot(self.weights, reward_vec))

        # 保存原始信息（后面评估用）
        info["reward_vector"] = reward_vec

        return obs, reward, terminated, truncated, info