r"""Bridge from this project's own `Benchmark` convention (MAXIMIZE, see
``plug_and_play/benchmarks.py``) to ``composite_ablation/solvers.py``'s
convention (vendored from https://github.com/tau315/composite-mobo, which
MINIMIZES throughout).

Only benchmarks with a composite structure (``Benchmark.raw_eval_fn`` set --
currently just ``composite_dtlz2``, see ``plug_and_play/benchmarks.py``) can
run through this module's composite solvers; direct-only benchmarks can
still run the direct solvers (``standard_mobo``, ``chebyshev_bo``,
``spherical_chebyshev_bo``) via ``to_minimize_convention``'s ``evaluate``
alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from plug_and_play.benchmarks import Benchmark


@dataclass
class MinimizeConventionProblem:
    r"""Everything `composite_ablation/solvers.py`'s solvers need, in their
    own (minimize) convention, derived from one `Benchmark`."""

    evaluate: "callable"
    dim: int
    ref_point: Tensor  # minimize convention: worse (larger) than every objective
    ideal: Tensor  # minimize convention: best achievable value per objective
    evaluate_components: Optional["callable"] = None
    compose: Optional["callable"] = None


def to_minimize_convention(
    bench: Benchmark, ideal: Optional[Tensor] = None
) -> MinimizeConventionProblem:
    r"""Wrap a MAXIMIZE-convention `Benchmark` into the MINIMIZE convention
    `composite_ablation/solvers.py` expects.

    Args:
        bench: a `Benchmark` from `plug_and_play.benchmarks.get_benchmark`.
        ideal: minimize-convention ideal point, required only for the
            Chebyshev-scalarization solvers (`chebyshev_bo`,
            `composite_chebyshev_bo`, `spherical_chebyshev_bo`,
            `composite_spherical_chebyshev_bo`). Defaults to all-zeros,
            correct for every DTLZ2-family benchmark (this project's
            `composite_dtlz2`/`dtlz2` included) -- DTLZ2's own raw
            objectives are non-negative with 0 exactly achieved on the
            Pareto front. Pass explicitly for any other benchmark.
    """
    evaluate = lambda X: -bench.eval_fn(X)  # noqa: E731
    ref_point = -torch.as_tensor(bench.ref_point, dtype=torch.double)
    ideal_t = (
        torch.zeros(bench.num_objectives, dtype=torch.double)
        if ideal is None
        else ideal.double()
    )

    evaluate_components = None
    compose = None
    if bench.raw_eval_fn is not None:
        # Benchmark.raw_eval_fn is already minimize-convention (see
        # plug_and_play/benchmarks.py's Benchmark docstring); only the
        # final reduction needs negating back to minimize convention.
        evaluate_components = bench.raw_eval_fn
        compose = lambda H: -bench.composite_reduction(H)  # noqa: E731

    return MinimizeConventionProblem(
        evaluate=evaluate,
        dim=bench.dim,
        ref_point=ref_point,
        ideal=ideal_t,
        evaluate_components=evaluate_components,
        compose=compose,
    )
