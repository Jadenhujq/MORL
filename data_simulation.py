import os
import numpy as np


def simulate_portfolio_data(
    T: int = 1000,
    N: int = 5,
    feature_dim: int = 10,
    seed: int = 42,
    save_path: str = "data",
    esg_mean: float = 0.5,
    esg_std: float = 0.1,
    return_scale: float = 0.001,
    cov_noise_scale: float = 0.02,
    feature_scale: float = 1.0,
):
    """
    构造时间序列投资组合数据，并保存到指定目录。

    参数说明
    ----------
    T : int
        时间步数量。
    N : int
        资产数量。
    feature_dim : int
        外生特征维度。
    seed : int
        随机数种子，便于复现。
    save_path : str
        保存 .npy 文件的目录。
    esg_mean : float
        ESG 分数均值。
    esg_std : float
        ESG 分数标准差。
    return_scale : float
        控制收益率的波动（越大，收益越波动）。
    cov_noise_scale : float
        控制协方差矩阵的随机扰动大小。
    feature_scale : float
        控制特征值范围（越大，特征数值范围越大）。
    """
    rng = np.random.default_rng(seed)

    # ==== 1. 基准协方差矩阵 ====
    # 构造一个正定矩阵作为长期协方差基准
    A = rng.normal(size=(N, N))
    base_cov = A @ A.T + np.eye(N) * 1e-3  # 添加小对角矩阵保持正定

    # ==== 2. 时间变化的协方差矩阵（T, N, N） ====
    cov_matrices = np.empty((T, N, N), dtype=np.float32)
    for t in range(T):
        # 每个时间步对 base_cov 施加随机扰动
        noise = rng.normal(scale=cov_noise_scale, size=(N, N))
        cov_t = base_cov + noise @ noise.T  # 保持对称正定
        cov_matrices[t] = cov_t.astype(np.float32)

    # ==== 3. 时间变化的均值 & ESG 分数 ====
    # ESG 分数随时间波动（T, N）
    esg_scores = rng.normal(loc=esg_mean, scale=esg_std, size=(T, N))
    esg_scores = np.clip(esg_scores, 0.0, 1.0)

    # 收益率均值受 ESG 制约，可根据需要调整关系
    base_mu = rng.uniform(0.0005, 0.002, size=N)
    mu_series = np.empty((T, N), dtype=np.float32)
    for t in range(T):
        # 这里简单假设 ESG 越高 → 调整收益（你可以换成更复杂的因子模型）
        esg_penalty = 0.0005 * (esg_scores[t] - esg_mean)
        mu_series[t] = (base_mu - esg_penalty).astype(np.float32)

    # ==== 4. 收益率时间序列（T, N） ====
    returns = np.empty((T, N), dtype=np.float32)
    for t in range(T):
        returns[t] = rng.multivariate_normal(mu_series[t], cov_matrices[t]).astype(np.float32)
    returns += rng.normal(scale=return_scale, size=(T, N)).astype(np.float32)

    # ==== 5. 外生特征（T, feature_dim） ====
    # 这里给出一个简单示例：AR(1) 过程
    features = np.empty((T, feature_dim), dtype=np.float32)
    phi = 0.8  # AR(1) 系数
    features[0] = rng.normal(scale=feature_scale, size=feature_dim)
    for t in range(1, T):
        noise = rng.normal(scale=feature_scale * (1 - phi), size=feature_dim)
        features[t] = phi * features[t - 1] + noise

    # ==== 6. 保存所有文件 ====
    os.makedirs(save_path, exist_ok=True)
    np.save(os.path.join(save_path, "returns.npy"), returns)
    np.save(os.path.join(save_path, "esg.npy"), esg_scores)
    np.save(os.path.join(save_path, "cov.npy"), cov_matrices)
    np.save(os.path.join(save_path, "features.npy"), features)


if __name__ == "__main__":
    simulate_portfolio_data()