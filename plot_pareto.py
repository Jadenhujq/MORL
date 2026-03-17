# plot_pareto.py

import numpy as np
import matplotlib.pyplot as plt

results = np.load("results.npy")

returns = results[:, 0]
risk = results[:, 1]
esg = results[:, 2]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(returns, risk, esg)

ax.set_xlabel("Return")
ax.set_ylabel("Risk")
ax.set_zlabel("ESG")

ax.set_title("Pareto Front (Portfolio MORL)")

plt.show()