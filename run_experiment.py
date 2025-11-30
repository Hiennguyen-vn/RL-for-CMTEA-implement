"""
run_experiment.py

Small driver to run RL_CMTEA on a Problems_py CMT* problem multiple times and save results.

Usage examples:
  python run_experiment.py --problem CMT1 --reps 20 --out results_CMT1

This will run RL_CMTEA with the default run parameters from the problem and save per-run
numpy arrays and a summary CSV in the output folder.
"""

import argparse
import os
import numpy as np
import csv
from datetime import datetime

from RL_CMTEA import RL_CMTEA
from run_cmt import build_tasks


def ensure_dir(d):
    os.makedirs(d, exist_ok=True)


def run_once(problem_name, seed, out_dir, override_pop=None, override_eva=None):
    tasks, run_para, pname = build_tasks(problem_name)
    sub_pop, sub_eva = run_para
    if override_pop:
        sub_pop = int(override_pop)
    if override_eva:
        sub_eva = int(override_eva)

    algo = RL_CMTEA(rng=int(seed))
    result = algo.run(tasks, (sub_pop, sub_eva))

    ts = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
    base = f"{pname}_seed{seed}_{ts}"
    conv_file = os.path.join(out_dir, base + "_conv.npy")
    conv_cv_file = os.path.join(out_dir, base + "_conv_cv.npy")
    bestx_file = os.path.join(out_dir, base + "_bestx.npy")
    np.save(conv_file, result['convergence'])
    np.save(conv_cv_file, result.get('convergence_cv', np.nan))
    np.save(bestx_file, np.array(result.get('bestX', []), dtype=object))

    best_costs = np.array(result['convergence'])[:, -1]
    best_cvs = np.array(result.get('convergence_cv', np.full_like(best_costs, np.nan)))[:, -1]

    rows = []
    for idx, (c, cv) in enumerate(zip(best_costs, best_cvs)):
        rows.append({
            'problem': pname,
            'task_idx': idx,
            'seed': seed,
            'best_cost': float(c),
            'best_cv': float(cv),
            'conv_file': conv_file,
            'conv_cv_file': conv_cv_file,
            'bestx_file': bestx_file,
        })
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--problems', nargs='+', default=['CMT1'], help='List of problem names, e.g. CMT1 CMT2')
    p.add_argument('--reps', type=int, default=5, help='Number of repetitions per problem (default: 5)')
    p.add_argument('--seed-start', type=int, default=0, help='Starting seed (default: 0)')
    p.add_argument('--out', default='results', help='Output folder')
    p.add_argument('--pop', type=int, help='Override population per task')
    p.add_argument('--eva', type=int, help='Override evaluation per task')
    args = p.parse_args()

    ensure_dir(args.out)
    summary_path = os.path.join(args.out, 'summary.csv')
    fieldnames = ['problem', 'task_idx', 'seed', 'best_cost', 'best_cv', 'conv_file', 'conv_cv_file', 'bestx_file']

    all_rows = []
    for prob in args.problems:
        prob_out = os.path.join(args.out, prob)
        ensure_dir(prob_out)
        for i in range(args.reps):
            seed = args.seed_start + i
            print(f'Running {prob} seed={seed} ({i+1}/{args.reps})')
            rows = run_once(prob, seed, prob_out, override_pop=args.pop, override_eva=args.eva)
            all_rows.extend(rows)

    with open(summary_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_rows:
            writer.writerow(r)

    print(f'Done. Summary written to {summary_path}. N rows={len(all_rows)}')


if __name__ == '__main__':
    main()
