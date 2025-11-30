#!/usr/bin/env python3
"""
main_py.py
Python reimplementation of the MATLAB `main.m` experiment driver.

Usage (defaults mirror main.m):
    python main_py.py

You can override with flags, e.g.:
    python main_py.py --reps 10 --algos FP_MFEA FP_GA --probs CMT1 CMT2 --save results.json

Notes:
- Problems are expected to be found in the `Problems_py` package (created earlier).
- Algorithms should be importable by name (module or class in Python); if an algorithm cannot be imported it will be skipped with a warning.
- Each algorithm must implement a `run(tasks, run_parameter_list)` method that returns a dict containing at least:
    - 'convergence': 2D list/array (history),
    - 'clock_time': numeric (seconds),
    - 'bestX': list/cell array (best solutions per run)

- Each problem class must implement `getTasks()` and `getRunParameterList()` (like the translated Problems_py classes).
- The script saves JSON results and will also attempt to save a .mat file if `scipy` is available.
"""

import argparse
import importlib
import json
import os
import sys
import time
from types import SimpleNamespace

try:
    import numpy as np
except Exception:
    np = None

try:
    from scipy.io import savemat
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)


def dynamic_import(name):
    """Try to import a module or class by name from several candidate locations."""
    # Try direct import
    try:
        mod = importlib.import_module(name)
        return mod
    except Exception:
        pass
    # Try Algorithms_py.<name>
    try:
        mod = importlib.import_module(f'Algorithms_py.{name}')
        return mod
    except Exception:
        pass
    # Try common subpackage locations inside Algorithms_py
    algo_candidates = [
        f'Algorithms_py.Single_task.GA.{name}',
        f'Algorithms_py.Single_task.{name}',
        f'Algorithms_py.Multi_task.MFEA.{name}',
        f'Algorithms_py.Multi_task.AT_MFEA.{name}',
        f'Algorithms_py.Multi_task.{name}',
    ]
    for cand in algo_candidates:
        try:
            mod = importlib.import_module(cand)
            return mod
        except Exception:
            continue
    # Try Problems_py.<name>
    try:
        mod = importlib.import_module(f'Problems_py.{name}')
        return mod
    except Exception:
        pass
    # Try Problems_py.Multi_task.Constrained_CMT.<name>
    try:
        mod = importlib.import_module(f'Problems_py.Multi_task.Constrained_CMT.{name}')
        return mod
    except Exception:
        pass
    return None


def instantiate_algo(name):
    """Instantiate an algorithm by name. Return instance or None if not found."""
    mod = dynamic_import(name)
    if mod is None:
        # maybe name is a class defined in a module with same name in cwd
        return None
    # If module contains a class with same name, instantiate it
    cls = getattr(mod, name, None)
    if callable(cls):
        try:
            return cls(name)
        except Exception:
            try:
                return cls()
            except Exception:
                return None
    # If module itself is callable (e.g. implemented as a function returning object)
    if callable(mod):
        try:
            return mod(name)
        except Exception:
            try:
                return mod()
            except Exception:
                return None
    # fallback: return module if it has run method
    if hasattr(mod, 'run'):
        return mod
    return None


def instantiate_problem(name):
    """Instantiate a problem from Problems_py.* packages. Return instance or raise."""
    # Try direct import from Problems_py packages
    candidates = [
        f'Problems_py.Multi_task.Constrained_CMT.{name}',
        f'Problems_py.{name}',
    ]
    for cand in candidates:
        try:
            mod = importlib.import_module(cand)
            cls = getattr(mod, name, None)
            if callable(cls):
                try:
                    return cls(name)
                except Exception:
                    return cls()
            # if module is callable
            if callable(mod):
                try:
                    return mod(name)
                except Exception:
                    return mod()
        except Exception:
            pass
    raise ImportError(f'Problem {name} not found in Problems_py packages')


def normalize_numpy(obj):
    """Convert numpy objects to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: normalize_numpy(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [normalize_numpy(v) for v in obj] 
    try:
        import numpy as _np
        if isinstance(obj, _np.ndarray):
            return obj.tolist()
        if isinstance(obj, (_np.float_, _np.float16, _np.float32, _np.float64)):
            return float(obj)
        if isinstance(obj, (_np.int_, _np.int8, _np.int16, _np.int32, _np.int64)):
            return int(obj)
    except Exception:
        pass
    return obj


def get_tasks_from_problem(p_obj):
    """Call a problem object's task accessor tolerantly.

    Tries common variants used across different ports: getTasks(), get_tasks().
    Raises AttributeError if none found.
    """
    for name in ('getTasks', 'get_tasks', 'getTask', 'get_task'):
        fn = getattr(p_obj, name, None)
        if callable(fn):
            return fn()
    raise AttributeError('Problem object has no getTasks/get_tasks method')


def get_run_parameter_list(p_obj):
    """Call a problem object's run-parameter accessor tolerantly.

    Tries getRunParameterList() and get_run_parameter_list().
    """
    for name in ('getRunParameterList', 'get_run_parameter_list', 'getRunParameters', 'get_run_parameters'):
        fn = getattr(p_obj, name, None)
        if callable(fn):
            return fn()
    if hasattr(p_obj, 'run_parameter_list'):
        return getattr(p_obj, 'run_parameter_list')
    raise AttributeError('Problem object has no getRunParameterList/get_run_parameter_list method')


def main(argv=None):
    parser = argparse.ArgumentParser(description='Python port of main.m experiment driver')
    parser.add_argument('--reps', type=int, default=30)
    parser.add_argument('--save', type=str, default='data_save.json')
    parser.add_argument('--algos', nargs='+', default=['FP_GA', 'FP_MFEA', 'FP_AT_MFEA'])
    parser.add_argument('--probs', nargs='+', default=['CMT1','CMT2','CMT3','CMT4','CMT5','CMT6','CMT7','CMT8','CMT9'])
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args(argv)

    reps = args.reps
    algo_names = args.algos
    prob_names = args.probs
    save_name = args.save
    seed = args.seed

    # instantiate algorithms
    algo_objs = {}
    for name in algo_names:
        obj = instantiate_algo(name)
        if obj is None:
            print(f'Warning: algorithm "{name}" could not be imported/instantiated. It will be skipped.')
        else:
            algo_objs[name] = obj

    # instantiate problems
    prob_objs = {}
    for name in prob_names:
        try:
            pobj = instantiate_problem(name)
            prob_objs[name] = pobj
        except Exception as e:
            print(f'Warning: problem "{name}" not found: {e}. Skipping.')

    # prepare result structure
    result = {p: {a: {'clock_time': 0.0, 'convergence': [], 'bestX': []} for a in algo_objs} for p in prob_objs}

    # run experiments
    base_seed = seed
    for rep in range(reps):
        print(f'Rep: {rep+1}/{reps} | seed={base_seed + rep}')
        # seed RNG for reproducibility
        try:
            import random
            random.seed(base_seed + rep)
            if np is not None:
                np.random.seed(base_seed + rep)
        except Exception:
            pass

        for p_name, p_obj in prob_objs.items():
            print(f'Problem: {p_name}')
            tasks = get_tasks_from_problem(p_obj)
            # Normalize tasks: Problems_py returns list of dicts; algorithms expect objects with attributes
            from types import SimpleNamespace
            normalized_tasks = []
            for t in tasks:
                if isinstance(t, dict):
                    ns = SimpleNamespace()
                    ns.dims = t.get('dims')
                    ns.fnc = t.get('fnc')
                    ns.Lb = t.get('Lb')
                    ns.Ub = t.get('Ub')
                    normalized_tasks.append(ns)
                else:
                    normalized_tasks.append(t)
            tasks = normalized_tasks
            run_param_list = get_run_parameter_list(p_obj)
            for a_name, a_obj in algo_objs.items():
                print(f'  Algorithm: {a_name} ...', flush=True)
                try:
                    start = time.time()
                    data = a_obj.run(tasks, run_param_list)
                    elapsed = time.time() - start
                except Exception as e:
                    print(f'    Error running {a_name} on {p_name}: {e}')
                    continue

                # expected keys: convergence, clock_time (or we'll set), bestX
                conv = data.get('convergence', data.get('convergence_history', []))
                clock = data.get('clock_time', elapsed)
                bestX = data.get('bestX', data.get('best_x', []))

                # store
                result[p_name][a_name]['clock_time'] += float(clock)
                result[p_name][a_name]['convergence'].append(normalize_numpy(conv))
                result[p_name][a_name]['bestX'].append(normalize_numpy(bestX))

                # pretty print best objective values if available
                try:
                    # if conv is a 2D array/list, take last column
                    if hasattr(conv, 'ndim') and getattr(conv, 'ndim') == 2:
                        last_col = conv[:, -1].tolist()
                    else:
                        # assume list of lists
                        last_col = [c[-1] for c in conv] if len(conv) and hasattr(conv[0], '__len__') else conv
                    print(f"    {a_name} Best Objective Values: {last_col}")
                except Exception:
                    print(f"    {a_name} run finished (cannot display convergence)")

        print('')

    # collect metadata and problem run_parameters
    data_save = {
        'reps': reps,
        'algo_cell': list(algo_objs.keys()),
        'prob_cell': list(prob_objs.keys()),
        'result': result,
        'save_time': time.asctime()
    }

    # add sub_pop and sub_eva and tasks_num_list per problem
    sub_pop = {}
    sub_eva = {}
    tasks_num_list = {}
    for p_name, p_obj in prob_objs.items():
        run_parameter_list = get_run_parameter_list(p_obj)
        sub_pop[p_name] = run_parameter_list[0]
        sub_eva[p_name] = run_parameter_list[1]
        tasks_num_list[p_name] = len(get_tasks_from_problem(p_obj))
    data_save['sub_pop'] = sub_pop
    data_save['sub_eva'] = sub_eva
    data_save['tasks_num_list'] = tasks_num_list

    # save JSON
    with open(save_name, 'w') as f:
        json.dump(normalize_numpy(data_save), f, indent=2)
    print(f'Results written to {save_name}')

    # optionally save MAT
    if SCIPY_AVAILABLE:
        mat_name = os.path.splitext(save_name)[0] + '.mat'
        try:
            savemat(mat_name, normalize_numpy(data_save))
            print(f'MAT file written to {mat_name}')
        except Exception as e:
            print(f'Could not save .mat: {e}')


if __name__ == '__main__':
    main()
