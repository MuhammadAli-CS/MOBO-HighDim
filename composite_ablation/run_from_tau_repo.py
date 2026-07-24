r"""Run tau315/composite-mobo's OWN code directly (cloned into
`composite_ablation/tau315_repo/`, not hand-vendored/reimplemented) --
every solver in that repo's `solvers.py`, including MORBO/composite_morbo,
against every one of that repo's own six benchmarks PLUS this project's
own five-objective/six-dimension DTLZ2 benchmark, with the eval budget
scaled by dimension.

Why clone instead of continuing to vendor (as `composite_ablation/
run_ablation.py`/`solvers.py`/`tau_benchmarks.py` did): that hand-vendored
copy already produced two real bugs from re-deriving each solver family's
own scaling knobs by hand (`--n-iter` silently ignored for the STCH
solvers, discovered only after burning cluster hours on it). Running his
own `solvers.py`/`benchmark_*.py` files directly sidesteps re-deriving
anything -- we drive his own functions with his own parameter names.

MORBO, unlike `run_ablation.py` (which excluded it -- this project
already has its own MORBO comparison), IS included here per direct
instruction, but using THIS PROJECT'S OWN `morbo` engine, not tau315's
own vendored copy (his repo has its own full `morbo/` port alongside
`solvers.py`, structurally near-identical to ours -- diffed directly,
the only difference in `run_one_replication.py` is one extra label our
own `tr_shape` work added). Achieved via Python's own import-caching
guarantee, not a fragile hack: `sys.modules` is checked before `sys.path`
for every `import X`/`from X import Y`, and lookups are cached by bare
top-level name. So importing OUR OWN `morbo.run_one_replication` FIRST,
before his `solvers.py` (which does `from morbo.run_one_replication
import run_one_replication` at its own module top level) ever gets
imported, guarantees his `batched_morbo`/`composite_batched_morbo`
transparently call OUR engine -- `import morbo` inside his module finds
"morbo" already in `sys.modules` (ours) and never touches his vendored
copy at all. Verified the two `run_one_replication.py`s are compatible
enough for this by diffing them directly before relying on it.

Eval budget scaling (dimension > 10 only; <=10 keeps his own fixed
45-eval default, already reasonable there): `n_init = clamp(round(dim/5),
10, 100)`, `remaining = clamp(8*dim, 80, 400)`. Each solver family spends
`remaining` in its OWN native parameterization (this is exactly the
knowledge `run_ablation.py` got wrong once already):
    standard_mobo / composite_mobo:            n_iter = remaining
    chebyshev_bo / composite_chebyshev_bo:      weights=8, per_weight = remaining/8
    spherical_chebyshev_bo / composite_..._bo:  weights=8, per_weight = remaining/8
    batched_morbo / composite_batched_morbo:    batch_size=5 (n_trust_regions),
                                                 n_iter = ceil(remaining/5)

Suite gating matches tau315's own protocol (his `benchmark_common.py`'s
`_solver_jobs`): "low"-suite benchmarks run standard/chebyshev/morbo;
"high"-suite (his own 50D/500D benchmarks) run spherical/morbo only --
running the plain-kernel solvers at 500D isn't a meaningful test (that
pathology is exactly what the spherical kernel exists to fix). This
project's own added 5-objective/6D DTLZ2 benchmark is "low"-suite (6D).

Usage:
    python -m composite_ablation.run_from_tau_repo --benchmark dtlz2_2obj_6d --pair standard --trials 20
    python -m composite_ablation.run_from_tau_repo --benchmark dtlz2_5obj_6d_ours --pair morbo --trials 20
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# MUST happen before importing tau315's solvers.py below -- see module
# docstring. Forces "morbo" into sys.modules as OUR package first.
import morbo.run_one_replication  # noqa: F401,E402

_TAU_REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tau315_repo")
if _TAU_REPO not in sys.path:
    sys.path.insert(0, _TAU_REPO)

# Compat shim, not an edit to the cloned file: tau315's solvers.py imports
# `OptimizationGradientError`, added to botorch after the version this
# project pins (botorch==0.9.5, see cluster/README.md's "Build the
# environment" section for why). Inject a stand-in before his module
# executes that import, so the clone stays byte-identical to upstream.
import botorch.exceptions.errors as _botorch_errors  # noqa: E402

if not hasattr(_botorch_errors, "OptimizationGradientError"):
    _botorch_errors.OptimizationGradientError = RuntimeError

import solvers as tau_solvers  # noqa: E402  (tau315's own solvers.py; now uses OUR morbo)
import benchmark_ackley_griewank_50d as _bm_ag50  # noqa: E402
import benchmark_ackley_griewank_6d as _bm_ag6  # noqa: E402
import benchmark_dtlz2 as _bm_dtlz2  # noqa: E402
import benchmark_five_ackley_6d as _bm_5ack  # noqa: E402
import benchmark_langermann_ackley_6d as _bm_lang  # noqa: E402
import benchmark_projected_langermann_500d as _bm_plang  # noqa: E402

from composite_ablation.adapters import MinimizeConventionProblem  # noqa: E402
from composite_ablation.run_ablation import run_pair  # noqa: E402  (reuse, don't re-derive)
from morbo.problems.composite_dtlz2_general import (  # noqa: E402
    composite_dtlz2_general_reduction,
    get_composite_dtlz2_general_fn,
)

TAU_PROBLEMS = {
    "dtlz2_2obj_6d": _bm_dtlz2.PROBLEM,
    "ackley_griewank_2obj_6d": _bm_ag6.PROBLEM,
    "ackley_griewank_2obj_50d": _bm_ag50.PROBLEM,
    "five_ackley_5obj_6d": _bm_5ack.PROBLEM,
    "langermann3_ackley_2obj_6d": _bm_lang.PROBLEM,
    "projected_langermann_2obj_500d": _bm_plang.PROBLEM,
}


def _make_our_dtlz2_5obj_6d() -> MinimizeConventionProblem:
    r"""This project's own DTLZ2, 5 objectives / 6 dimensions -- the same
    construction `plug_and_play/benchmarks.py`'s `composite_dtlz2` uses
    (`morbo/problems/composite_dtlz2_general.py`), wrapped into tau315's
    own `evaluate`/`evaluate_components`/`compose` minimize-convention
    interface directly (bypassing `plug_and_play.Benchmark`'s MAXIMIZE
    convention + `composite_ablation/adapters.py`'s flip, since here we
    want tau315's own convention from the start). `ideal=0`/`ref_point=6`
    (minimize convention) match this project's own established DTLZ2
    convention (`ref_point=[-6]*M` in the MAXIMIZE convention used
    throughout `experiments/dtlz2_5obj_6d/`).
    """
    dim, num_objectives = 6, 5
    raw_f, bounds = get_composite_dtlz2_general_fn(dim=dim, num_objectives=num_objectives)
    lb, ub = bounds[0].double(), bounds[1].double()

    def evaluate_components(X: torch.Tensor) -> torch.Tensor:
        X_native = lb + X.double() * (ub - lb)
        return raw_f(X_native)  # already minimize-convention

    def compose(H: torch.Tensor) -> torch.Tensor:
        return -composite_dtlz2_general_reduction(H, num_objectives=num_objectives)

    return MinimizeConventionProblem(
        evaluate=lambda X: compose(evaluate_components(X)),
        dim=dim,
        ref_point=torch.full((num_objectives,), 6.0, dtype=torch.double),
        ideal=torch.zeros(num_objectives, dtype=torch.double),
        evaluate_components=evaluate_components,
        compose=compose,
    )


def _eval_budget(dim: int) -> tuple:
    r"""(n_init, remaining) -- see module docstring's scaling rule."""
    if dim <= 10:
        return 5, 40
    n_init = min(max(round(dim / 5), 10), 100)
    remaining = min(max(8 * dim, 80), 400)
    return n_init, remaining


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--benchmark",
        choices=list(TAU_PROBLEMS) + ["dtlz2_5obj_6d_ours"],
        required=True,
    )
    parser.add_argument(
        "--pair", choices=["standard", "chebyshev", "spherical", "morbo"], required=True,
        help="one pair per job -- see cluster/submit_tau_repo_full.sh.",
    )
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--weights", type=int, default=8)
    parser.add_argument("--quick", action="store_true", help="tiny smoke-test-scale run")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--num-threads", type=int, default=8)
    parser.add_argument("--num-interop-threads", type=int, default=4)
    args = parser.parse_args()

    torch.set_num_threads(args.num_threads)
    torch.set_num_interop_threads(args.num_interop_threads)

    if args.benchmark == "dtlz2_5obj_6d_ours":
        problem = _make_our_dtlz2_5obj_6d()
        num_objectives, suite = 5, "low"
    else:
        tau_bench = TAU_PROBLEMS[args.benchmark]
        problem = MinimizeConventionProblem(
            evaluate=tau_bench.evaluate,
            dim=tau_bench.dim,
            ref_point=tau_bench.ref_point,
            ideal=tau_bench.ideal,
            evaluate_components=tau_bench.evaluate_components,
            compose=tau_bench.compose,
        )
        num_objectives, suite = tau_bench.num_objectives, tau_bench.suite

    if args.pair in ("standard", "chebyshev") and suite != "low":
        raise ValueError(f"{args.pair} is only meaningful on a 'low'-suite benchmark (got suite={suite!r})")
    if args.pair == "spherical" and suite != "high":
        raise ValueError(f"spherical is only meaningful on a 'high'-suite benchmark (got suite={suite!r})")

    n_init, remaining = _eval_budget(problem.dim)
    if args.quick:
        args.trials, n_init, remaining = 1, 3, 4

    print(
        f"benchmark={args.benchmark} dim={problem.dim} num_objectives={num_objectives} "
        f"suite={suite} pair={args.pair} n_init={n_init} remaining_budget={remaining} "
        f"total_evals={n_init + remaining}",
        flush=True,
    )

    if args.pair == "standard":
        run_pair(
            "standard_mobo / composite_mobo",
            tau_solvers.standard_mobo, tau_solvers.composite_mobo, problem,
            trials=args.trials, seed0=args.seed,
            solver_kwargs=dict(n_init=n_init, n_iter=remaining),
            out_dir=args.out_dir,
        )
    elif args.pair == "chebyshev":
        weights = tau_solvers.simplex_weights(args.weights, num_objectives, seed=314159)
        per_weight = max(1, round(remaining / args.weights))
        run_pair(
            "chebyshev_bo / composite_chebyshev_bo",
            tau_solvers.chebyshev_bo, tau_solvers.composite_chebyshev_bo, problem,
            trials=args.trials, seed0=args.seed,
            solver_kwargs=dict(weights=weights, n_init=n_init, n_per_scalarization=per_weight),
            out_dir=args.out_dir,
        )
    elif args.pair == "spherical":
        weights = tau_solvers.simplex_weights(args.weights, num_objectives, seed=314159)
        per_weight = max(1, round(remaining / args.weights))
        run_pair(
            "spherical_chebyshev_bo / composite_spherical_chebyshev_bo",
            tau_solvers.spherical_chebyshev_bo, tau_solvers.composite_spherical_chebyshev_bo, problem,
            trials=args.trials, seed0=args.seed,
            solver_kwargs=dict(weights=weights, n_init=n_init, n_per_scalarization=per_weight),
            out_dir=args.out_dir,
        )
    else:  # morbo -- uses OUR morbo engine, see module docstring
        batch_size = 5
        n_iter = max(1, math.ceil(remaining / batch_size))
        run_pair(
            "batched_morbo / composite_batched_morbo",
            tau_solvers.batched_morbo, tau_solvers.composite_batched_morbo, problem,
            trials=args.trials, seed0=args.seed,
            solver_kwargs=dict(n_init=n_init, n_iter=n_iter, batch_size=batch_size),
            out_dir=args.out_dir,
        )


if __name__ == "__main__":
    main()
