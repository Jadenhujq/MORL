import numpy as np
import os


def simulate_portfolio_data(
    T=1000,
    N=5,
    seed=42,
    save_path="data"
):
    np.random.seed(seed)

    # ===== 1. 协方差矩阵 =====
    A = np.random.randn(N, N)
    cov = A @ A.T

    # ===== 2. 平均收益 =====
    mu = np.random.uniform(0.0005, 0.002, size=N)

    # ===== 3. ESG =====
    esg = np.random.uniform(0, 1, size=N)

    # ===== 4. ESG trade-off =====
    mu = mu - 0.001 * esg

    # ===== 5. 生成 returns =====
    returns = np.random.multivariate_normal(mu, cov, size=T)

    # ===== 6. 保存 =====
    os.makedirs(save_path, exist_ok=True)

    np.save(f"{save_path}/returns.npy", returns)
    np.save(f"{save_path}/esg.npy", esg)

    print("Data saved to:", save_path)


if __name__ == "__main__":
    simulate_portfolio_data()