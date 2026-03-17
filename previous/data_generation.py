# data_generation.py
import numpy as np

def generate_synthetic_market(
    n_assets=6,
    n_steps=2000,
    risk_free_rate=0.0001,
    seed=42
):
    rng = np.random.default_rng(seed)
    k = 3  # financial factors
    l = 2  # ESG factors

    A_f = 0.8 * np.eye(k) + 0.05 * rng.standard_normal((k, k))
    A_e = 0.6 * np.eye(l) + 0.05 * rng.standard_normal((l, l))

    F = np.zeros((n_steps, k))
    E = np.zeros((n_steps, l))
    for t in range(1, n_steps):
        F[t] = A_f @ F[t-1] + 0.01 * rng.standard_normal(k)
        E[t] = A_e @ E[t-1] + 0.02 * rng.standard_normal(l)

    B_f = rng.normal(0, 0.2, size=(n_assets, k))
    B_e = rng.normal(0, 0.3, size=(n_assets, l))
    idio_sigma = rng.uniform(0.005, 0.02, size=n_assets)

    returns = np.zeros((n_steps, n_assets + 1))
    esg_scores = np.zeros((n_steps, n_assets + 1))
    returns[:, 0] = 1.0 + risk_free_rate
    esg_scores[:, 0] = 0.0

    for t in range(n_steps):
        mu_t = B_f @ F[t]
        ret_t = mu_t + rng.normal(0, idio_sigma)
        returns[t, 1:] = np.exp(ret_t)
        esg_scores[t, 1:] = np.clip(B_e @ E[t] + 0.5, 0, 1)

    window = 60
    log_ret = np.log(returns[:, 1:])
    rolling_mean = np.vstack([
        np.mean(log_ret[max(0, t-window):t+1], axis=0)
        for t in range(n_steps)
    ])
    rolling_vol = np.vstack([
        np.std(log_ret[max(0, t-window):t+1], axis=0)
        for t in range(n_steps)
    ])

    features = {
        "rolling_mean": rolling_mean,
        "rolling_vol": rolling_vol,
        "factors": F,
        "esg_factors": E,
    }
    return returns, esg_scores, features

if __name__ == "__main__":
    R, E, feats = generate_synthetic_market()
    print("Synthetic data generated:")
    print("Returns shape:", R.shape)
    print("ESG shape:", E.shape)
    for k, v in feats.items():
        print(f"{k} shape:", v.shape)
