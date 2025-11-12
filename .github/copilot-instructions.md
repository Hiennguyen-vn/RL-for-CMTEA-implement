## Purpose
Short, actionable guidance for AI coding agents working on this evolutionary optimization repository (MATLAB->Python ports). Focus on conventions, entry points, and patterns to avoid wrong edits.

## Big-picture architecture
- Top-level scripts and modules are small, focused: `RL_CMTEA.py` (main algorithm), `KT.py` (knowledge transfer / block DE + kmeans), `SPX.py` (SBX & mutation), `DE_*.py` (DE operators), `initializeMP.py` (population init & task interface), `selectMP.py` (selection / ranking). `run_experiment.py` is a runnable demo.
- Data flow: Tasks (objects with `.dim` or `.dims` and `.evaluate(x)->(cost, cv)`) -> `initialize_mp` produces per-task populations (list-of-lists) -> `RL_CMTEA.run` loops: KT -> operators (SBX/DE) -> evaluate -> select_mp -> update Q-table/convergence.

## Key conventions and invariants (must preserve)
- All decision vectors (`rnvec`) are normalized to [0,1]. Operators always clip to [0,1] (see `DE_*`, `SPX.py`, `KT.py`). Never assume real-valued scaling; if you add conversion, implement `uni2real` and keep normalization for operators.
- Population layout: `population` is a list of per-task lists: population[task_index][individual_index]. Many functions expect this shape (see `KT.py`, `initializeMP.py`).
- Best individual position: several operators assume the best individual is at index 0 of a population (see `DE_best_1.py` where `best = population[0]`). Preserve or update callers consistently.
- Selection API: `select_mp(population, offspring, bestobj, bestCV, bestX, ep)` returns `(population_new, rank, bestobj, bestCV, bestX, Flag)` — maintain this signature when refactoring.
- RNG usage: Prefer `np.random.default_rng(rng)` or passing `rng` to functions; do not mix global `np.random.seed` unpredictably. Many modules accept `rng` (see `KT.py`, `SPX.py`).

## File-level notes & examples (search these when making changes)
- `RL_CMTEA.py`: orchestrates training loop, Q-learning + UCB, uses `KT`, `SBX`, `DE_*`, `initialize_mp`, `select_mp`. Example run: `python run_experiment.py` (calls `RL_CMTEA.run`).
- `KT.py`: builds `corre` (MATLAB 1-based indices) then decodes blocks, runs kmeans (sklearn optional) and applies DE on blocks. Note `RL_CMTEA_USE_SKLEARN` env var to switch to sklearn KMeans. Be careful with 1-based/0-based index translations in `corre_decode`.
- `initializeMP.py`: shows Task interface (see `Task.evaluate` example). Use `initialize_mp(Individual_factory, pop_size, Tasks, dims, init_type=...)` to set up populations.
- `selectMP.py`: uses Epsilon-Constraint, Feasible Priority, Stochastic Ranking; returns rank arrays and updates best (watch comparisons and epsilon thresholds).
- `SPX.py` (SBX & mutation): SBX implementation mirrors Deb (2001). Use `SBX(params, population)` which returns offspring keeping population ordering.

## Developer workflows & commands
- Quick demo: python run_experiment.py (runs `RL_CMTEA` demo with `initializeMP.Task` example tasks).
- There is no test harness included. To validate changes locally: run small `if __name__=='__main__'` blocks already present (many modules contain small demos). For algorithm changes, run `run_experiment.py` with small `pop` and `eva` values to verify end-to-end behavior.

## Patterns & pitfalls to watch for
- In-place vs copy: several functions deep-copy populations (e.g., operators return deep copies). When changing behavior, preserve deep-copy semantics or clearly document mutation.
- MATLAB port idioms: many comments and logic mirror MATLAB (1-based indices). Look for `+1 / -1` translations, esp. in `KT.py` and `corre` handling.
- Population length assumptions: some logic computes halves or expects even sizes; tests/demos sometimes use odd `n` to exercise edge-cases (see `SPX.py` demo). Keep logic robust for odd/even n.

## Integration & external deps
- Optional: `scikit-learn` for KMeans. Toggle via env var `RL_CMTEA_USE_SKLEARN=1` (default falls back to NumPy implementation in `KT.py`).

## How to ask for edits (prompt templates for AI)
- When changing an operator: "Modify DE_rand_1.py to use adaptive F per-generation; ensure `rnvec` remains clipped to [0,1] and update any callers that assume deterministic mutation; run `run_experiment.py` with pop=10,eva=200 to sanity-check."
- When refactoring population layout: "Refactor population to use numpy arrays instead of lists; update `initialize_mp`, `KT.corre_decode`, `select_mp`, and all operator modules; include safe deep-copy semantics and a small demo run to validate results." 

## Minimal checklist for PRs touching algorithms
- Preserve normalization [0,1] on `rnvec`.
- Update/verify `select_mp` signature and that `population[0]` still represents best if expected.
- Run `run_experiment.py` small demo to ensure no runtime errors.
- If enabling sklearn KMeans, document env var and add try/except fallback (already present in `KT.py`).

If any section is unclear or you'd like more examples (e.g., suggested tests or a small runner script), tell me which area and I will iterate.
