"""Load results saved by bench_run.py and produce plots (PNG) summarizing convergence.

Usage example:
  python plot_results.py --summary results/summary.csv --out figures/

Produces per-problem convergence plot (mean ± std) and per-run traces.
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--summary', required=True, help='Path to summary.csv produced by bench_run.py')
    p.add_argument('--out', default='figures', help='Output folder for PNGs')
    return p.parse_args()


def ensure_dir(d):
    os.makedirs(d, exist_ok=True)


def load_convergence_files(df_row):
    conv = np.load(df_row['conv_file'])
    conv_cv = np.load(df_row['conv_cv_file'])
    return conv, conv_cv


def plot_for_problem(df, out_dir):
    # df: rows for one problem (multiple seeds and task_idx combined)
    problem = df['problem'].iloc[0]
    tasks = sorted(df['task_idx'].unique())
    ensure_dir(out_dir)

    for t in tasks:
        df_t = df[df['task_idx'] == t]
        all_conv = []
        for _, row in df_t.iterrows():
            conv = np.load(row['conv_file'])
            # conv shape: (n_tasks, n_gens); we select column for this task
            if conv.shape[0] <= t:
                continue
            vals = conv[t, :]
            all_conv.append(vals)

        if not all_conv:
            print(f"No convergence arrays for {problem} task {t}")
            continue

        # pad sequences to same length
        maxlen = max(len(v) for v in all_conv)
        arr = np.vstack([np.pad(v, (0, maxlen - len(v)), constant_values=np.nan) for v in all_conv])

        mean = np.nanmean(arr, axis=0)
        std = np.nanstd(arr, axis=0)

        fig, ax = plt.subplots(figsize=(8,4))
        x = np.arange(mean.size)
        ax.plot(x, mean, label='mean')
        ax.fill_between(x, mean-std, mean+std, alpha=0.3, label='±1 std')
        ax.set_title(f"{problem} - Task {t} convergence")
        ax.set_xlabel('generation')
        ax.set_ylabel('best objective')
        ax.legend()
        figfile = os.path.join(out_dir, f"{problem}_task{t}_conv.png")
        fig.savefig(figfile, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved {figfile}")


def main():
    args = parse_args()
    ensure_dir(args.out)
    df = pd.read_csv(args.summary)

    for problem, dfp in df.groupby('problem'):
        plot_for_problem(dfp, args.out)


if __name__ == '__main__':
    main()
