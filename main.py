# main.py

import os
import subprocess
import sys

SCRIPTS = [
    ("Step 1: Data Simulation", "data_simulation.py"),
    ("Step 2: Train PPO (Multi-Weights)", "train_PPO.py"),
    ("Step 3: Evaluate Models", "evaluate.py"),
    ("Step 4: Plot Pareto Front", "plot_pareto.py"),
]

def run_script(name, script):
    print("\n" + "=" * 50)
    print(name)
    print("=" * 50)

    if not os.path.exists(script):
        print(f"Error: script not found -> {script}")
        sys.exit(1)

    result = subprocess.run([sys.executable, script])

    if result.returncode != 0:
        print(f"Error occurred while running {script}")
        sys.exit(1)

    print(f"Finished: {script}")

def main():
    print("\nMORL Portfolio Pipeline Starting...\n")

    results_exist = os.path.exists("results.npy")

    for name, script in SCRIPTS:

        # 如果结果已存在，跳过训练和评估
        if results_exist and script in ["train_PPO.py", "evaluate.py"]:
            print(f"Skipping {script} (results already exist)")
            continue

        run_script(name, script)

    print("\nAll steps completed successfully.\n")

if __name__ == "__main__":
    main()