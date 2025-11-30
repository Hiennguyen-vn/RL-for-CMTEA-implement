#!/usr/bin/env python3
"""Visualize results produced by main_py.py (JSON format).

Usage:
  python3 results_visualize.py --input results_50reps.json --out figs

Generates one PNG per problem with subplots per task showing mean +/- std convergence curves.
"""
import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt


def pad_and_stack(vecs):
    # vecs: list of list/1D arrays (may contain nan)
    maxlen = max(len(v) for v in vecs)
    arr = np.full((len(vecs), maxlen), np.nan, dtype=float)
    for i, v in enumerate(vecs):
        arr[i, :len(v)] = v
    return arr


def plot_problem(problem_name, pdata, out_dir):
    # pdata: dict of algorithm -> list of per-rep convergence arrays
    algos = sorted(pdata.keys())
    # Determine number of tasks from any algo (they should match)
    sample_algo = algos[0] if algos else None
    if sample_algo is None:
        print(f"No algorithms to plot for {problem_name}")
        return
    first_rep = None
    for a in algos:
        if len(pdata[a]):
            first_rep = pdata[a][0]
            break
    if first_rep is None:
        print(f"No data for problem {problem_name}")
        return
    num_tasks = len(first_rep)

    # Layout
    ncols = 3
    nrows = int(np.ceil(num_tasks / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for t in range(num_tasks):
        ax = axes[t // ncols][t % ncols]
        for a_idx, a in enumerate(algos):
            # collect rep-wise convergence for task t
            rep_convs = []
            for rep in pdata[a]:
                try:
                    conv = np.array(rep[t], dtype=float)
                except Exception:
                    continue
                rep_convs.append(conv)
            if not rep_convs:
                continue
            arr = pad_and_stack(rep_convs)
            mean = np.nanmean(arr, axis=0)
            std = np.nanstd(arr, axis=0)
            x = np.arange(mean.size)
            c = colors[a_idx % len(colors)]
            ax.plot(x, mean, label=a, color=c)
            ax.fill_between(x, mean - std, mean + std, color=c, alpha=0.2)
        ax.set_title(f'Task {t+1}')
        ax.set_xlabel('Generation')
        ax.set_ylabel('Objective')
        ax.grid(True, linestyle='--', alpha=0.4)
        if t == 0:
            ax.legend(loc='best', fontsize='small')

    # Turn off unused axes
    for idx in range(num_tasks, nrows * ncols):
        ax = axes[idx // ncols][idx % ncols]
        ax.axis('off')

    fig.suptitle(problem_name)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_path = os.path.join(out_dir, f'{problem_name}_convergence.png')
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'Wrote {out_path}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', default='results_50reps.json')
    p.add_argument('--out', default='figs')
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print('Loading', args.input)
    with open(args.input, 'r') as f:
        data = json.load(f)

    result = data.get('result', {})
    for prob, prob_data in result.items():
        # prob_data: algo -> {'convergence': [rep arrays], 'bestX': ...}
        pdata = {}
        for algo, ad in prob_data.items():
            convs = ad.get('convergence', [])
            pdata[algo] = convs
        plot_problem(prob, pdata, args.out)


if __name__ == '__main__':
    main()
