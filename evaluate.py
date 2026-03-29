import numpy as np
from stable_baselines3 import PPO
from portfolio_env import PortfolioEnv
from wrapper import ScalarizationWrapper

WEIGHTS_LIST = [
    [0.7, 0.2, 0.1],
    [0.5, 0.3, 0.2],
    [0.3, 0.4, 0.3],
    [0.2, 0.3, 0.5],
    [0.8, 0.1, 0.1],
]

results = []

returns = np.load("data/returns.npy")
esg = np.load("data/esg.npy")

for i, weights in enumerate(WEIGHTS_LIST):
    print(f"\nEvaluating model {i}")

    env = PortfolioEnv(returns, esg)
    env = ScalarizationWrapper(env, weights)

    model = PPO.load(f"models/ppo_w{i}")

    obs, _ = env.reset()
    done = False

    total_return = 0
    total_risk = 0
    total_esg = 0
    steps = 0

    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, _, info = env.step(action)

        total_return += info["return"]
        total_risk += info["risk"]
        total_esg += info["esg"]

        steps += 1

    avg_return = total_return / steps
    avg_risk = total_risk / steps
    avg_esg = total_esg / steps

    print(f"Return={avg_return:.3f}, Risk={avg_risk:.3f}, ESG={avg_esg:.3f}")

    results.append([avg_return, avg_risk, avg_esg])

# 保存结果
np.save("results.npy", np.array(results))
print("\nSaved results.npy")