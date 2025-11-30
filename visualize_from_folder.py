#!/usr/bin/env python3
"""Visualize results stored in a results folder produced by run_experiment.py

It reads <root>/summary.csv and the conv .npy files referenced there. For each problem
it creates a PNG with one subplot per task. Each seed/run is drawn as a faint line and
the mean across runs is drawn as a bold line.

Usage:
  python3 visualize_from_folder.py --root results_CMT_all --out results_CMT_all/figs
"""
import argparse
import os
import csv
import numpy as np
import matplotlib.pyplot as plt


def load_summary(root):
    path = os.path.join(root, 'summary.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(f'summary.csv not found in {root}')
    rows = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def group_by_problem(rows):
    d = {}
    for r in rows:
        p = r['problem']
        d.setdefault(p, []).append(r)
    return d


def pad_and_stack(vecs):
    maxlen = max(v.shape[-1] for v in vecs)
    arr = np.full((len(vecs), maxlen), np.nan, dtype=float)
    for i, v in enumerate(vecs):
        arr[i, : v.shape[-1]] = v
    return arr


def plot_problem(problem, rows, out_dir):
    # rows: list of summary rows for this problem
    # collect conv files per seed
    conv_files = sorted({r['conv_file'] for r in rows})
    if not conv_files:
        print('No conv files for', problem)
        return
    # load all conv arrays
    convs = []
    for f in conv_files:
        if not os.path.exists(f):
            print('Missing file', f)
            continue
        arr = np.load(f)
        # ensure 2D: tasks x gens
        arr = np.asarray(arr)
        if arr.ndim == 1:
            arr = arr[np.newaxis, :]
        convs.append(arr)
    if not convs:
        print('No valid conv arrays for', problem)
        return
    # number of tasks is first dim
    num_tasks = convs[0].shape[0]
    ncols = 3
    nrows = int(np.ceil(num_tasks / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)

    for t in range(num_tasks):
        ax = axes[t // ncols][t % ncols]
        task_runs = []
        for arr in convs:
            if t >= arr.shape[0]:
                continue
            task_runs.append(np.asarray(arr[t], dtype=float))
        if not task_runs:
            continue
        stacked = pad_and_stack([r for r in task_runs])
        # plot individual runs
        for r in stacked:
            ax.plot(np.arange(r.size), r, color='gray', alpha=0.3, linewidth=0.8)
        mean = np.nanmean(stacked, axis=0)
        ax.plot(np.arange(mean.size), mean, color='C0', linewidth=2.0, label='mean')
        ax.set_title(f'Task {t+1}')
        ax.set_xlabel('Generation')
        ax.set_ylabel('Objective')
        ax.grid(True, linestyle='--', alpha=0.4)
        if t == 0:
            ax.legend()

    for idx in range(num_tasks, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis('off')

    fig.suptitle(problem)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'{problem}_linechart.png')
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print('Wrote', out_path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--out', required=False)
    args = p.parse_args()
    root = args.root
    out_dir = args.out or os.path.join(root, 'figs')
    rows = load_summary(root)
    byp = group_by_problem(rows)
    for prob, rlist in byp.items():
        plot_problem(prob, rlist, out_dir)


if __name__ == '__main__':
    main()
