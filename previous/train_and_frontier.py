# train_and_frontier.py
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from previous.data_generation import generate_synthetic_market
from previous.esg_env import ESGPortfolioEnv


def pareto_filter(points):
    n = points.shape[0]
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        if dominated[i]:
            continue
        dominates = np.all(points >= points[i], axis=1) & np.any(points > points[i], axis=1)
        dominated |= dominates
        dominated[i] = False
    return ~dominated


def make_env(weight_tuple):
    def _init():
        returns, esg_scores, features = generate_synthetic_market()
        env = ESGPortfolioEnv(
            returns=returns,
            esg_scores=esg_scores,
            features=features
        )
        env.weights_obj = weight_tuple  # custom attribute if needed
        return env
    return _init


def train_single_agent(weight_tuple, timesteps=100_000):
    env = DummyVecEnv([make_env(weight_tuple)])
    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=timesteps)
    return model


def evaluate_agent(model, weight_tuple, episodes=10):
    returns, esg_scores, features = generate_synthetic_market(seed=999)
    env = ESGPortfolioEnv(returns, esg_scores, features)
    stats = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        episode_rewards = []
        esg_vals, risk_vals, log_returns = [], [], []
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action, weights_obj=weight_tuple)
            done = terminated or truncated
            episode_rewards.append(reward)
            log_returns.append(info["log_return"])
            risk_vals.append(info["risk_term"])
            esg_vals.append(info["esg_term"])
        stats.append((
            np.mean(log_returns),
            np.mean(risk_vals),
            np.mean(esg_vals)
        ))
    return np.mean(stats, axis=0)


def main():
    weight_list = [
        (0.7, 0.2, 0.1),
        (0.6, 0.3, 0.1),
        (0.5, 0.3, 0.2),
        (0.4, 0.3, 0.3),
        (0.3, 0.3, 0.4),
    ]
    results = []
    for w in weight_list:
        print(f"Training for weights {w} ...")
        model = train_single_agent(w, timesteps=50_000)
        metrics = evaluate_agent(model, w)
        print(f"  Metrics (return, risk, ESG): {metrics}")
        results.append(metrics)

    results = np.array(results)
    mask = pareto_filter(results)
    print("\nNon-dominated points:")
    for w, m in zip(weight_list, results):
        print(f"Weights {w}: Metrics {m}")
    print("\nPareto-front indices:", np.where(mask)[0])


if __name__ == "__main__":
    main()
