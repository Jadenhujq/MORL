import numpy as np


def generate_weight_grid(n_points=20):
    """
    在 3D simplex 上生成均匀网格
    w1 + w2 + w3 = 1, wi >= 0
    """
    weights = []

    for i in range(1, n_points):
        for j in range(1, n_points - i):
            k = n_points - i - j
            if k < 1:
                continue  

            w1 = i / n_points
            w2 = j / n_points
            w3 = k / n_points

            weights.append([w1, w2, w3])

    return weights


if __name__ == "__main__":
    weights = generate_weight_grid(n_points=20)

    print(f"Generated {len(weights)} weight vectors:\n")
    for w in weights:
        print(w)