import argparse
import numpy as np

from RL_CMTEA import RL_CMTEA
from Problems_py.Multi_task.Constrained_CMT.CMT1 import CMT1
from Problems_py.Multi_task.Constrained_CMT.CMT2 import CMT2
from Problems_py.Multi_task.Constrained_CMT.CMT3 import CMT3
from Problems_py.Multi_task.Constrained_CMT.CMT4 import CMT4
from Problems_py.Multi_task.Constrained_CMT.CMT5 import CMT5
from Problems_py.Multi_task.Constrained_CMT.CMT6 import CMT6
from Problems_py.Multi_task.Constrained_CMT.CMT7 import CMT7
from Problems_py.Multi_task.Constrained_CMT.CMT8 import CMT8
from Problems_py.Multi_task.Constrained_CMT.CMT9 import CMT9


class CMTTaskAdapter:
    """Wrap one task dict from CMT* classes into the interface RL_CMTEA expects."""

    def __init__(self, fnc, lb, ub):
        self.fnc = fnc
        self.lb = np.asarray(lb, dtype=float)
        self.ub = np.asarray(ub, dtype=float)
        self.dim = self.lb.size
        self.dims = self.dim

    def evaluate(self, rnvec):
        real_x = self.lb + np.asarray(rnvec, dtype=float) * (self.ub - self.lb)
        obj, cv = self.fnc(real_x)
        return float(obj), float(cv)


CMT_PROBLEMS = {
    "CMT1": CMT1,
    "CMT2": CMT2,
    "CMT3": CMT3,
    "CMT4": CMT4,
    "CMT5": CMT5,
    "CMT6": CMT6,
    "CMT7": CMT7,
    "CMT8": CMT8,
    "CMT9": CMT9,
}


def build_tasks(problem_name):
    problem_name = problem_name.upper()
    if problem_name not in CMT_PROBLEMS:
        raise ValueError(f"Unknown problem {problem_name}. Pick one of {sorted(CMT_PROBLEMS)}")

    problem = CMT_PROBLEMS[problem_name]()
    task_dicts = problem.get_tasks()
    tasks = [CMTTaskAdapter(t["fnc"], t["Lb"], t["Ub"]) for t in task_dicts]
    sub_pop, sub_eva = problem.get_run_parameter_list()
    return tasks, (sub_pop, sub_eva), problem_name


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run RL-CMTEA on a constrained multitask benchmark (CMT1-9)."
    )
    parser.add_argument(
        "-p", "--problem", default="CMT1", help="Problem name in Problems_py/Multi_task/Constrained_CMT (default: CMT1)."
    )
    parser.add_argument("--sub-pop", type=int, help="Override population size per task.")
    parser.add_argument("--sub-eva", type=int, help="Override evaluation budget per task.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for RL_CMTEA (default: 0).")
    return parser.parse_args()


def main():
    args = parse_args()
    tasks, run_para, name = build_tasks(args.problem)
    sub_pop, sub_eva = run_para
    if args.sub_pop:
        sub_pop = int(args.sub_pop)
    if args.sub_eva:
        sub_eva = int(args.sub_eva)

    algo = RL_CMTEA(rng=args.seed)
    result = algo.run(tasks, (sub_pop, sub_eva))

    best_costs = result["convergence"][:, -1]
    best_cvs = result["convergence_cv"][:, -1]

    print(f"Problem: {name} | sub_pop={sub_pop} | sub_eva={sub_eva} | seed={args.seed}")
    for idx, (cost, cv) in enumerate(zip(best_costs, best_cvs)):
        status = "feasible" if cv <= 0 else f"cv={cv:.3e}"
        print(f"Task {idx + 1}: best cost {cost:.6f} ({status})")

    print("\nDecision vectors (real space) for the best individuals:")
    for idx, (vec_norm, task) in enumerate(zip(result["bestX"], tasks)):
        real_vec = task.lb + vec_norm * (task.ub - task.lb)
        prefix = np.array2string(real_vec[: min(10, real_vec.size)], precision=4, max_line_width=120)
        suffix = "" if real_vec.size <= 10 else f"... total_dim={real_vec.size}"
        print(f"- Task {idx + 1}: {prefix} {suffix}")


if __name__ == "__main__":
    main()
