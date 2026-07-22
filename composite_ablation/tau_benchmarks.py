r"""The six composite benchmark problems, vendored from a collaborator's repo.

Source: https://github.com/tau315/composite-mobo (`benchmark_common.py` +
each `benchmark_*.py`), by tau315. Vendored (not reimplemented) so this
project's own solver-vs-composite ablation (`composite_ablation/
run_ablation.py`) can run against exactly the same benchmark suite that
repo itself uses, in addition to this project's own `composite_dtlz2`
(`plug_and_play/benchmarks.py`).

Kept in the source repo's own convention throughout: MINIMIZATION on
`[0, 1]^d`, `evaluate_components(X) -> H`, `compose(H) -> Y`, with
`compose(evaluate_components(X)) == evaluate(X)` -- the same convention
`composite_ablation/solvers.py` (also vendored from that repo) expects, so
no adapter is needed here (contrast `composite_ablation/adapters.py`,
needed only to bridge THIS project's own MAXIMIZE-convention `Benchmark`
objects into that convention).

`suite` mirrors the source repo's own low/high split
(`benchmark_common.py`'s `_solver_jobs`): "low"-suite benchmarks there run
its qLogEHVI and (non-spherical) Chebyshev solver pairs; "high"-suite
benchmarks run its spherical-linear Chebyshev pair (plus its own MORBO,
which this project excludes -- see `composite_ablation/solvers.py`'s
module docstring for why). `composite_ablation/run_ablation.py` uses this
field the same way.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
import torch

Tensor = torch.Tensor
Evaluator = Callable[[Tensor], Tensor]
Composer = Callable[[Tensor], Tensor]
Suite = Literal["low", "high"]

ACKLEY_UPPER_BOUND = 20.0 + np.e - np.exp(-1.0)


@dataclass(frozen=True)
class TauBenchmarkProblem:
    """A deterministic composite multi-objective minimization problem
    (trimmed transcription of the source repo's own `BenchmarkProblem`;
    validation/plotting-only fields dropped -- `composite_ablation/
    run_ablation.py` does its own hypervolume tracing and reporting)."""

    name: str
    slug: str
    dim: int
    num_objectives: int
    suite: Suite
    evaluate_components: Evaluator
    compose: Composer
    ideal: Tensor
    ref_point: Tensor

    def evaluate(self, X: Tensor) -> Tensor:
        return self.compose(self.evaluate_components(X))


def orthogonal_matrix(dim: int, seed: int) -> Tensor:
    """Return a fixed deterministic dense orthogonal matrix."""

    generator = torch.Generator().manual_seed(seed)
    raw = torch.randn(dim, dim, generator=generator, dtype=torch.double)
    q, r = torch.linalg.qr(raw)
    signs = torch.where(torch.diagonal(r) >= 0, 1.0, -1.0)
    return q * signs


def orthonormal_rows(rows: int, dim: int, seed: int) -> Tensor:
    """Return `rows` fixed dense orthonormal directions in R^dim."""

    if rows > dim:
        raise ValueError("the number of rows cannot exceed the dimension")
    generator = torch.Generator().manual_seed(seed)
    raw = torch.randn(dim, rows, generator=generator, dtype=torch.double)
    q, r = torch.linalg.qr(raw, mode="reduced")
    signs = torch.where(torch.diagonal(r) >= 0, 1.0, -1.0)
    return (q * signs).T.contiguous()


def transformed_inputs(X: Tensor, center: Tensor, rotation: Tensor, scale: float) -> Tensor:
    """Shift, densely rotate, and scale points from the unit cube."""

    return scale * ((X.double() - center.double()) @ rotation.double().T)


def ackley_components(Z: Tensor) -> Tensor:
    """The two canonical intermediate quantities of the Ackley function."""

    return torch.stack(
        (Z.square().mean(dim=-1), torch.cos(2.0 * torch.pi * Z).mean(dim=-1)),
        dim=-1,
    )


def compose_ackley(H: Tensor, *, normalize: bool = True) -> Tensor:
    """Apply the known Ackley outer map to two intermediate quantities."""

    mean_square = H[..., 0].clamp_min(0)
    mean_cosine = H[..., 1].clamp(-1.0, 1.0)
    value = (
        -20.0 * torch.exp(-0.2 * mean_square.sqrt()) - torch.exp(mean_cosine) + 20.0 + torch.e
    )
    return value / ACKLEY_UPPER_BOUND if normalize else value


def griewank_components(Z: Tensor) -> Tensor:
    """The quadratic and cosine-product intermediates of Griewank."""

    indices = torch.arange(1, Z.shape[-1] + 1, dtype=Z.dtype, device=Z.device)
    return torch.stack(
        (Z.square().sum(dim=-1) / 4000.0, torch.cos(Z / indices.sqrt()).prod(dim=-1)),
        dim=-1,
    )


def compose_griewank(H: Tensor, upper_bound: float) -> Tensor:
    """Apply and safely normalize the known Griewank outer map."""

    quadratic = H[..., 0].clamp_min(0)
    cosine_product = H[..., 1].clamp(-1.0, 1.0)
    return (1.0 + quadratic - cosine_product) / upper_bound


def griewank_upper_bound(center: Tensor, scale: float) -> float:
    """A conservative bound over the unit cube, invariant to rotation."""

    furthest_squared = torch.maximum(center.square(), (1.0 - center).square()).sum()
    return float(2.0 + scale**2 * furthest_squared / 4000.0)


def compose_langermann(H: Tensor, coefficients: Tensor, targets: Tensor = None) -> Tensor:
    """Normalized minimization form of the generalized Langermann outer map."""

    coefficients = coefficients.to(dtype=H.dtype, device=H.device)
    distances = (
        H.clamp_min(0)
        if targets is None
        else (H - targets.to(dtype=H.dtype, device=H.device)).square()
    )
    raw = -(
        coefficients * torch.exp(-distances / torch.pi) * torch.cos(torch.pi * distances)
    ).sum(dim=-1)
    total = coefficients.sum()
    return (raw + total) / (2.0 * total)


# ---------------------------------------------------------------------------
# benchmark_dtlz2.py
# ---------------------------------------------------------------------------
def _dtlz2_components(X: Tensor) -> Tensor:
    X = X.double()
    distance = (X[..., 1:] - 0.5).square().sum(dim=-1)
    angle = torch.pi * X[..., 0] / 2.0
    return torch.stack((distance, angle.cos(), distance, angle.sin()), dim=-1)


def _dtlz2_compose(H: Tensor) -> Tensor:
    distance_1 = H[..., 0].clamp_min(0)
    angle_1 = H[..., 1].clamp(0.0, 1.0)
    distance_2 = H[..., 2].clamp_min(0)
    angle_2 = H[..., 3].clamp(0.0, 1.0)
    return torch.stack(((1.0 + distance_1) * angle_1, (1.0 + distance_2) * angle_2), dim=-1)


DTLZ2_2OBJ_6D = TauBenchmarkProblem(
    name="DTLZ2 (2 objectives, 6 dimensions)",
    slug="dtlz2_2obj_6d",
    dim=6,
    num_objectives=2,
    suite="low",
    evaluate_components=_dtlz2_components,
    compose=_dtlz2_compose,
    ideal=torch.zeros(2, dtype=torch.double),
    ref_point=torch.full((2,), 2.5, dtype=torch.double),
)


# ---------------------------------------------------------------------------
# benchmark_ackley_griewank_6d.py
# ---------------------------------------------------------------------------
def _make_ackley_griewank(dim: int, ackley_c: float, griewank_c: float,
                           ackley_scale: float, griewank_scale: float,
                           seed_a: int, seed_g: int, suite: Suite, slug: str) -> TauBenchmarkProblem:
    ackley_center = torch.full((dim,), ackley_c, dtype=torch.double)
    griewank_center = torch.full((dim,), griewank_c, dtype=torch.double)
    ackley_rotation = orthogonal_matrix(dim, seed=seed_a)
    griewank_rotation = orthogonal_matrix(dim, seed=seed_g)
    griewank_upper = griewank_upper_bound(griewank_center, griewank_scale)

    def evaluate_components(X: Tensor) -> Tensor:
        ackley_z = transformed_inputs(X, ackley_center, ackley_rotation, ackley_scale)
        griewank_z = transformed_inputs(X, griewank_center, griewank_rotation, griewank_scale)
        return torch.cat((ackley_components(ackley_z), griewank_components(griewank_z)), dim=-1)

    def compose(H: Tensor) -> Tensor:
        return torch.stack(
            (compose_ackley(H[..., 0:2]), compose_griewank(H[..., 2:4], griewank_upper)), dim=-1
        )

    return TauBenchmarkProblem(
        name=f"Ackley versus Griewank (2 objectives, {dim} dimensions)",
        slug=slug,
        dim=dim,
        num_objectives=2,
        suite=suite,
        evaluate_components=evaluate_components,
        compose=compose,
        ideal=torch.zeros(2, dtype=torch.double),
        ref_point=torch.full((2,), 2.5, dtype=torch.double),
    )


ACKLEY_GRIEWANK_2OBJ_6D = _make_ackley_griewank(
    dim=6, ackley_c=0.25, griewank_c=0.75, ackley_scale=8.0, griewank_scale=12.0,
    seed_a=6101, seed_g=6102, suite="low", slug="ackley_griewank_2obj_6d",
)
ACKLEY_GRIEWANK_2OBJ_50D = _make_ackley_griewank(
    dim=50, ackley_c=0.35, griewank_c=0.65, ackley_scale=5.0, griewank_scale=2.5,
    seed_a=6501, seed_g=6502, suite="high", slug="ackley_griewank_2obj_50d",
)


# ---------------------------------------------------------------------------
# benchmark_five_ackley_6d.py
# ---------------------------------------------------------------------------
def _make_five_ackley() -> TauBenchmarkProblem:
    dim, num_objectives, scale = 6, 5, 8.0
    simplex = torch.eye(num_objectives, dtype=torch.double)
    simplex = simplex - simplex.mean(dim=0, keepdim=True)
    simplex = simplex / simplex.norm(dim=-1, keepdim=True)
    centers = torch.full((num_objectives, dim), 0.5, dtype=torch.double)
    centers[:, :num_objectives] += 0.28 * simplex
    rotations = tuple(orthogonal_matrix(dim, seed=6200 + o) for o in range(num_objectives))

    def evaluate_components(X: Tensor) -> Tensor:
        groups = []
        for o in range(num_objectives):
            z = transformed_inputs(X, centers[o], rotations[o], scale)
            groups.append(ackley_components(z))
        return torch.cat(groups, dim=-1)

    def compose(H: Tensor) -> Tensor:
        return torch.stack(
            [compose_ackley(H[..., 2 * i : 2 * i + 2]) for i in range(num_objectives)], dim=-1
        )

    return TauBenchmarkProblem(
        name="Five shifted Ackley objectives (5 objectives, 6 dimensions)",
        slug="five_ackley_5obj_6d",
        dim=dim,
        num_objectives=num_objectives,
        suite="low",
        evaluate_components=evaluate_components,
        compose=compose,
        ideal=torch.zeros(num_objectives, dtype=torch.double),
        ref_point=torch.full((num_objectives,), 2.5, dtype=torch.double),
    )


FIVE_ACKLEY_5OBJ_6D = _make_five_ackley()


# ---------------------------------------------------------------------------
# benchmark_langermann_ackley_6d.py
# ---------------------------------------------------------------------------
def _make_langermann_ackley() -> TauBenchmarkProblem:
    dim = 6
    langermann_centers = torch.tensor(
        [
            [0.15, 0.20, 0.25, 0.30, 0.35, 0.40],
            [0.30, 0.25, 0.20, 0.15, 0.40, 0.35],
            [0.20, 0.35, 0.15, 0.40, 0.25, 0.30],
        ],
        dtype=torch.double,
    )
    langermann_coefficients = torch.tensor([1.0, 2.0, 3.0], dtype=torch.double)
    ackley_center = torch.full((dim,), 0.80, dtype=torch.double)
    ackley_rotation = orthogonal_matrix(dim, seed=6301)
    ackley_scale = 8.0

    def evaluate_components(X: Tensor) -> Tensor:
        X = X.double()
        distances = (X.unsqueeze(-2) - langermann_centers).square().sum(dim=-1)
        ackley_z = transformed_inputs(X, ackley_center, ackley_rotation, ackley_scale)
        return torch.cat((distances, ackley_components(ackley_z)), dim=-1)

    def compose(H: Tensor) -> Tensor:
        return torch.stack(
            (
                compose_langermann(H[..., 0:3], langermann_coefficients),
                compose_ackley(H[..., 3:5]),
            ),
            dim=-1,
        )

    return TauBenchmarkProblem(
        name="Langermann-3 versus Ackley (2 objectives, 6 dimensions)",
        slug="langermann3_ackley_2obj_6d",
        dim=dim,
        num_objectives=2,
        suite="low",
        evaluate_components=evaluate_components,
        compose=compose,
        ideal=torch.zeros(2, dtype=torch.double),
        ref_point=torch.full((2,), 2.5, dtype=torch.double),
    )


LANGERMANN3_ACKLEY_2OBJ_6D = _make_langermann_ackley()


# ---------------------------------------------------------------------------
# benchmark_projected_langermann_500d.py
# ---------------------------------------------------------------------------
def _make_projected_langermann_500d() -> TauBenchmarkProblem:
    dim = 500
    projections = orthonormal_rows(7, dim, seed=500_007)
    coefficients_1 = torch.tensor([1.0, 1.5, 2.0, 2.5], dtype=torch.double)
    coefficients_2 = torch.tensor([1.0, 1.25, 1.5, 1.75, 2.0], dtype=torch.double)
    targets_1 = torch.tensor([-0.65, -0.35, 0.25, 0.55], dtype=torch.double)
    targets_2 = torch.tensor([0.65, 0.35, -0.45, 0.05, 0.50], dtype=torch.double)

    def evaluate_components(X: Tensor) -> Tensor:
        projected = (2.0 * X.double() - 1.0) @ projections.T
        objective_1 = projected[..., [0, 1, 2, 3]]
        objective_2 = projected[..., [0, 1, 4, 5, 6]]
        return torch.cat((objective_1, objective_2), dim=-1)

    def compose(H: Tensor) -> Tensor:
        return torch.stack(
            (
                compose_langermann(H[..., 0:4], coefficients_1, targets_1),
                compose_langermann(H[..., 4:9], coefficients_2, targets_2),
            ),
            dim=-1,
        )

    return TauBenchmarkProblem(
        name="Projected Langermann (2 objectives, 500 dimensions; 4+5 components)",
        slug="projected_langermann_2obj_500d",
        dim=dim,
        num_objectives=2,
        suite="high",
        evaluate_components=evaluate_components,
        compose=compose,
        ideal=torch.zeros(2, dtype=torch.double),
        ref_point=torch.full((2,), 2.5, dtype=torch.double),
    )


PROJECTED_LANGERMANN_2OBJ_500D = _make_projected_langermann_500d()


TAU_BENCHMARKS: dict = {
    "dtlz2_2obj_6d": DTLZ2_2OBJ_6D,
    "ackley_griewank_2obj_6d": ACKLEY_GRIEWANK_2OBJ_6D,
    "ackley_griewank_2obj_50d": ACKLEY_GRIEWANK_2OBJ_50D,
    "five_ackley_5obj_6d": FIVE_ACKLEY_5OBJ_6D,
    "langermann3_ackley_2obj_6d": LANGERMANN3_ACKLEY_2OBJ_6D,
    "projected_langermann_2obj_500d": PROJECTED_LANGERMANN_2OBJ_500D,
}


def get_tau_benchmark(slug: str) -> TauBenchmarkProblem:
    if slug not in TAU_BENCHMARKS:
        raise ValueError(f"Unknown tau benchmark {slug!r}. Available: {sorted(TAU_BENCHMARKS)}")
    return TAU_BENCHMARKS[slug]
