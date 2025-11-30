"""Run specified algorithm modules directly on Problems (uses run_cmt.build_tasks to build Tasks).

Usage:
  python run_algos_direct.py --algos Algorithms_py.Single_task.GA.FP_GA Algorithms_py.Multi_task.MFEA.FP_MFEA --problem CMT1 --seed 1

This script imports each algorithm module, instantiates the class whose name equals the last path component
(e.g. module ...FP_GA -> class FP_GA), calls run(tasks, run_para) and prints + saves results to temp JSON.
"""

import argparse
import importlib
import json
import os
import time
from run_cmt import build_tasks
from types import SimpleNamespace
from datetime import datetime


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--algos', nargs='+', required=True)
    p.add_argument('--problem', default='CMT1')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', default='temp_algo_results.json')
    return p.parse_args()


def instantiate_from_module_path(path):
    mod = importlib.import_module(path)
    cls_name = path.split('.')[-1]
    cls = getattr(mod, cls_name, None)
    if callable(cls):
        try:
            return cls()
        except Exception:
            try:
                return cls(cls_name)
            except Exception:
                return None
    # fallback: if module itself has run function, return module
    if hasattr(mod, 'run'):
        return mod
    return None


def normalize(obj):
    # recursively normalize numpy arrays to python lists
    try:
        import numpy as _np
    except Exception:
        _np = None
    if _np is not None:
        if isinstance(obj, _np.ndarray):
            return obj.tolist()
        if isinstance(obj, (list, tuple)):
            return [normalize(o) for o in obj]
        if isinstance(obj, dict):
            return {k: normalize(v) for k, v in obj.items()}
        if isinstance(obj, (_np.floating, _np.float32, _np.float64)):
            return float(obj)
        if isinstance(obj, (_np.integer,)):
            return int(obj)
    # fallback for basic types
    if isinstance(obj, (list, tuple)):
        return [normalize(o) for o in obj]
    if isinstance(obj, dict):
        return {k: normalize(v) for k, v in obj.items()}
    return obj


def main():
    args = parse_args()
    tasks, run_para, pname = build_tasks(args.problem)
    # build normalized tasks as run_cmt does
    norm_tasks = []
    for t in tasks:
        # algorithms expect tasks either as dicts with keys 'dims','fnc','Lb','Ub'
        # or objects with attributes 'dims','fnc','Lb','Ub'. Convert accordingly.
        if isinstance(t, dict):
            ns = SimpleNamespace()
            ns.dims = t.get('dims')
            ns.fnc = t.get('fnc')
            ns.Lb = t.get('Lb')
            ns.Ub = t.get('Ub')
            norm_tasks.append(ns)
        else:
            # t may be a CMTTaskAdapter with .dims, .fnc, .lb, .ub
            ns = SimpleNamespace()
            ns.dims = getattr(t, 'dims', getattr(t, 'dim', None))
            ns.fnc = getattr(t, 'fnc', None)
            # support both .Lb/.Ub or .lb/.ub
            ns.Lb = getattr(t, 'Lb', getattr(t, 'lb', None))
            ns.Ub = getattr(t, 'Ub', getattr(t, 'ub', None))
            norm_tasks.append(ns)

    results = {}
    for algo_path in args.algos:
        print(f'Running algo {algo_path} on {pname} (seed={args.seed})')
        algo_obj = instantiate_from_module_path(algo_path)
        if algo_obj is None:
            print('  Could not instantiate', algo_path)
            continue
        try:
            start = time.time()
            data = algo_obj.run(norm_tasks, run_para)
            elapsed = time.time() - start
        except Exception as e:
            print('  Error during run:', e)
            continue

        # save summary
        results[algo_path] = {
            'clock_time': float(data.get('clock_time', elapsed)),
            'convergence': normalize(data.get('convergence', [])),
            'bestX': normalize(data.get('bestX', [])),
        }

    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2)
    print('Wrote', args.out)


if __name__ == '__main__':
    main()
