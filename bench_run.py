"""Run RL-CMTEA across tasks and seeds and save results for plotting.

Usage examples:
  python bench_run.py --problem CMT1 --seeds 1 2 3 --out results/
  python bench_run.py --problems CMT1 CMT2 --seeds 1 2 --seeds 1 2 3 --out results/ --pop 40 --eva 2000

This script saves per-run numpy arrays and a summary CSV with columns:
  problem, task_idx, seed, best_cost, best_cv, conv_file, conv_cv_file, bestx_file

It intentionally uses modest defaults and exposes CLI flags to override.
"""

import argparse
import os
import csv
import numpy as np
from datetime import datetime

from run_cmt import build_tasks
from RL_CMTEA import RL_CMTEA


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--problems", nargs='+', default=['CMT1'], help="Problem names (CMT1..CMT9)")
    p.add_argument("--seeds", nargs='+', type=int, default=[0], help="List of seeds to run")
    p.add_argument("--out", default='results', help="Output folder")
    p.add_argument("--pop", type=int, help="Override sub_pop per task")
    p.add_argument("--eva", type=int, help="Override sub_eva per task")
    p.add_argument("--sklearn", action='store_true', help="Use sklearn KMeans via env var")
    return p.parse_args()


def ensure_dir(d):
    os.makedirs(d, exist_ok=True)


def run_problem(problem_name, seeds, out_dir, override_pop=None, override_eva=None):
    tasks, run_para, pname = build_tasks(problem_name)
    sub_pop, sub_eva = run_para
    if override_pop:
        sub_pop = int(override_pop)
    if override_eva:
        sub_eva = int(override_eva)

    summary_rows = []
    meta_dir = os.path.join(out_dir, pname)
    ensure_dir(meta_dir)

    for seed in seeds:
        print(f"Running {pname} seed={seed} pop={sub_pop} eva={sub_eva}")
        algo = RL_CMTEA(rng=int(seed))
        result = algo.run(tasks, (sub_pop, sub_eva))

        ts = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
        # Save numpy arrays
        base = f"{pname}_seed{seed}_{ts}"
        conv_file = os.path.join(meta_dir, base + "_conv.npy")
        conv_cv_file = os.path.join(meta_dir, base + "_conv_cv.npy")
        bestx_file = os.path.join(meta_dir, base + "_bestx.npy")
        np.save(conv_file, result['convergence'])
        np.save(conv_cv_file, result['convergence_cv'])
        # bestX can be list of arrays: save as object dtype
        np.save(bestx_file, np.array(result['bestX'], dtype=object))

        best_costs = result['convergence'][:, -1]
        best_cvs = result['convergence_cv'][:, -1]
        for idx, (cost, cv) in enumerate(zip(best_costs, best_cvs)):
            summary_rows.append({
                'problem': pname,
                'task_idx': idx,
                'seed': seed,
                'best_cost': float(cost),
                'best_cv': float(cv),
                'conv_file': conv_file,
                'conv_cv_file': conv_cv_file,
                'bestx_file': bestx_file,
            })

    return summary_rows


def main():
    args = parse_args()
    if args.sklearn:
        os.environ['RL_CMTEA_USE_SKLEARN'] = '1'

    ensure_dir(args.out)
    summary_path = os.path.join(args.out, 'summary.csv')
    fieldnames = ['problem', 'task_idx', 'seed', 'best_cost', 'best_cv', 'conv_file', 'conv_cv_file', 'bestx_file']

    all_rows = []
    for prob in args.problems:
        rows = run_problem(prob, args.seeds, args.out, override_pop=args.pop, override_eva=args.eva)
        all_rows.extend(rows)

    # Write summary CSV
    with open(summary_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_rows:
            writer.writerow(r)

    print(f"Done. Summary written to {summary_path}. N rows={len(all_rows)}")


if __name__ == '__main__':
    main()
