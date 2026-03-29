import os
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO

from portfolio_env import DynamicPortfolioEnv
from generate_weight import generate_weight_grid


# ===== 1. 加载数据 =====
returns = np.load("data/returns.npy")
esg_scores = np.load("data/esg.npy")
cov_matrices = np.load("data/cov.npy")
features = np.load("data/features.npy")


# ===== 2. 生成 weights（核心）=====
weights_list = generate_weight_grid(n_points=3)


# ===== 3. 创建模型保存目录 =====
os.makedirs("models", exist_ok=True)


# ===== 4. 训练循环 =====
for i, (rho, zeta, phi) in enumerate(weights_list):

    print("\n" + "="*50)
    print(f"Training Model {i}")
    print(f"Weights: rho={rho:.2f}, zeta={zeta:.2f}, phi={phi:.2f}")
    print("="*50)

    # ===== 创建环境 =====
    env = DynamicPortfolioEnv(
        returns=returns,
        esg_scores=esg_scores,
        cov_matrices=cov_matrices,
        features=features,
        rho=rho,
        zeta=zeta,
        phi=phi
    )

    # ===== 创建 PPO =====
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="./logs/",
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
    )

    # ===== 训练 =====
    model.learn(total_timesteps=20000)

    # ===== 保存模型 =====
    model_path = f"models/ppo_{i}"
    model.save(model_path)

    print(f"Saved to {model_path}")


print("\nAll models trained successfully.\n")