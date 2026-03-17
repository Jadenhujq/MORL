import os
import numpy as np
from stable_baselines3 import PPO
from env import PortfolioEnv
from wrapper import ScalarizationWrapper

# ========= 可调参数 =========
WEIGHTS_LIST = [
    [0.7, 0.2, 0.1],
    [0.5, 0.3, 0.2],
    [0.3, 0.4, 0.3],
    [0.2, 0.3, 0.5],
    [0.8, 0.1, 0.1],
]

TIMESTEPS = 300_000
SAVE_DIR = "models"

os.makedirs(SAVE_DIR, exist_ok=True)

# ========= 读取数据 =========
returns = np.load("data/returns.npy")
esg = np.load("data/esg.npy")

# ========= 训练循环 =========
for i, weights in enumerate(WEIGHTS_LIST):
    print(f"\n===== Training model {i} with weights {weights} =====")

    env = PortfolioEnv(returns, esg)
    env = ScalarizationWrapper(env, weights)

    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=TIMESTEPS)

    model_path = f"{SAVE_DIR}/ppo_w{i}"
    model.save(model_path)

    print(f"Saved to {model_path}")