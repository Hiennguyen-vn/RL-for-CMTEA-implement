"""
Minimal demo to run RL_CMTEA on two simple constrained Sphere tasks.

Usage:
    python run_experiment.py
"""

import numpy as np

from RL_CMTEA import RL_CMTEA
from initializeMP import Task as DemoTask


def main():
    # Two tasks with different dimensionalities and the demo evaluate() from initializeMP.Task
    tasks = [
        DemoTask(dim=20),
        DemoTask(dim=35),
    ]

    # (population size per task, evaluation budget per task)
    run_para = (40, 2000)

    algo = RL_CMTEA(rng=1234)
    result = algo.run(tasks, run_para)

    best_costs = result["convergence"][:, -1]
    best_cvs = result["convergence_cv"][:, -1]

    for idx, (cost, cv) in enumerate(zip(best_costs, best_cvs)):
        status = "feasible" if cv <= 0 else f"cv={cv:.3e}"
        print(f"Task {idx + 1}: best cost {cost:.6f} ({status})")

    print("\nDecision vectors (normalized) for the best individuals:")
    for idx, vec in enumerate(result["bestX"]):
        print(f"- Task {idx + 1}: {np.array2string(vec, precision=4, max_line_width=120)}")


if __name__ == "__main__":
    main()
