"""Two-objective problems that vary only the structure of one outer map.

The structure under test:

        x --> h0 --------------> g1 --> f1        g1 is what we vary
          \
           -> h1,h2,h3,h4 -----> g2 --> f2        g2 is FIXED (Ackley)

Both objectives are minimised.  The composite model observes all five
intermediates (h0..h4); the direct model sees only (f1, f2).

Everything except g1 is held fixed across conditions -- the input dimension,
the five inner functions (same random Fourier draws), and g2 -- so any
difference in the composite advantage is attributable to the structure of g1
alone.  The three choices mirror the single-objective study:

    identity  g1(z) = z        linear: f1 is itself a GP, so the direct model
                               is correctly specified and nothing is lost
    cube      g1(z) = z^3      bijective but nonlinear: nothing is lost, but
                               the direct model now has the wrong shape
    square    g1(z) = z^2      not bijective: the direct model has the wrong
                               shape AND cannot recover the sign of h0

Reading across those three separates the two channels in the multi-objective
setting the way the matched pairs did in the single-objective one: the
identity-to-cube step is misspecification with no information change, and the
cube-to-square step adds information loss at comparable nonlinearity.

Objectives are affinely normalised to roughly [0, 1] on a fixed reference set,
because hypervolume is not scale-free and g1(z)=z^2 and g1(z)=z^3 otherwise
span wildly different ranges.  The normalisation is part of the known g for
both arms, so it gives neither an advantage.
"""
from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass
from typing import Callable, Dict

import torch
from torch import Tensor
from torch.quasirandom import SobolEngine

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from three_arm.test_functions import RFFDraw                      # noqa: E402

DTYPE = torch.double
N_INNER = 5                      # h0 for g1, h1..h4 for g2


def ackley(Z: Tensor, a: float = 20.0, b: float = 0.2, c: float = 2 * math.pi) -> Tensor:
    """Standard Ackley on the last dimension; minimised at the origin."""
    m = Z.shape[-1]
    s1 = (Z ** 2).sum(-1) / m
    s2 = torch.cos(c * Z).sum(-1) / m
    return -a * torch.exp(-b * s1.sqrt()) - torch.exp(s2) + a + math.e


OUTER_G1: Dict[str, Callable[[Tensor], Tensor]] = {
    "identity": lambda z: z,
    "sinh": torch.sinh,
    "cube": lambda z: z ** 3,
    "square": lambda z: z ** 2,
}
# sinh is the second bijective-but-nonlinear map, included because z^3 turned
# out to be a poor partner for z^2: its dynamic range is large enough that a
# few seeds dominate the mean (max +131% against a median of +23%), which
# swamped the contrast it was meant to carry.  sinh is monotone and nonlinear
# with a tamer tail over the range h reaches.
# Whether g1 can be inverted on the range h0 actually reaches.  Both identity
# and cube are strictly increasing; the square folds at the origin.
G1_INJECTIVE = {"identity": True, "sinh": True, "cube": True, "square": False}


@dataclass
class TwoStageProblem:
    name: str
    dim: int
    evaluate: Callable[[Tensor], Tensor]
    evaluate_components: Callable[[Tensor], Tensor]
    compose: Callable[[Tensor], Tensor]
    ref_point: Tensor
    ideal: Tensor


def shared_ref_point(dim: int = 6, lengthscale: float = 0.5, draw_seed: int = 0,
                     pad: float = 0.05, g2: str = "ackley",
                     kernel: str = "matern") -> Tensor:
    """One reference point for every g1, so hypervolumes are comparable.

    Hypervolume is measured against a reference point, so letting each
    condition pick its own would make the three numbers incommensurable --
    exactly the comparison the experiment is about.  This takes the
    element-wise worst over the conditions.
    """
    worst = None
    for name in OUTER_G1:
        p = make_two_stage(name, dim, lengthscale, draw_seed, _ref_point=False,
                           g2=g2, kernel=kernel)
        ref = SobolEngine(dim, scramble=True, seed=31337).draw(8192).to(DTYPE)
        y = p.evaluate(ref).max(dim=0).values
        worst = y if worst is None else torch.maximum(worst, y)
    return worst + pad


def make_two_stage(g1_name: str, dim: int = 6, lengthscale: float = 0.5,
                   draw_seed: int = 0, n_ref: int = 32768,
                   _ref_point: bool = True, g2: str = "ackley",
                   kernel: str = "matern") -> TwoStageProblem:
    """g2 = "ackley" uses four intermediates and a hard multimodal map;
    g2 = "identity" uses one intermediate and the identity.

    The identity variant is the cleaner experiment.  Ackley is itself strongly
    nonlinear and strongly non-injective, so Composite gains a great deal on
    the second objective in *every* condition -- a constant baseline that
    dilutes whatever g1 contributes, and the reason the linear condition still
    showed +17% there.  With the identity, f2 = h2 is exactly a Gaussian
    process, Direct is correctly specified on that objective and Composite
    gains nothing from it, so the whole advantage is attributable to g1 and
    the absolute numbers mean something rather than only the contrasts.
    """
    g1 = OUTER_G1[g1_name]
    n_inner = N_INNER if g2 == "ackley" else 2
    # Matern 5/2 by default, because that is BoTorch's default kernel: drawing
    # h from the same family the surrogate assumes is what makes the model
    # correctly specified about the inner function, so any remaining
    # misspecification is attributable to g.
    draws = [RFFDraw(dim, lengthscale, seed=1000 * draw_seed + j, kernel=kernel)
             for j in range(n_inner)]

    def components(X: Tensor) -> Tensor:
        return torch.stack([d(X) for d in draws], dim=-1)          # (n, n_inner)

    def raw(C: Tensor) -> Tensor:
        f1 = g1(C[..., 0])
        f2 = ackley(C[..., 1:]) if g2 == "ackley" else C[..., 1]
        return torch.stack([f1, f2], dim=-1)                       # (n, 2)

    # Percentile normalisation per objective, on a fixed reference set.
    ref = SobolEngine(dim, scramble=True, seed=31337).draw(n_ref).to(DTYPE)
    R = raw(components(ref))
    lo = torch.quantile(R, 0.01, dim=0)
    hi = torch.quantile(R, 0.99, dim=0)
    span = (hi - lo).clamp_min(1e-12)

    def compose(C: Tensor) -> Tensor:
        return (raw(C) - lo) / span

    def evaluate(X: Tensor) -> Tensor:
        return compose(components(X))

    # Reference point worse than essentially everything, in the minimise
    # convention the solvers use.  Taken from the same reference set so it is
    # identical in spirit across conditions, then padded.
    normed = compose(components(ref))
    ideal = normed.min(dim=0).values
    ref_point = (shared_ref_point(dim, lengthscale, draw_seed, g2=g2, kernel=kernel)
                 if _ref_point
                 else normed.max(dim=0).values + 0.05)

    return TwoStageProblem(
        name=f"two_stage_{g1_name}_{g2}", dim=dim, evaluate=evaluate,
        evaluate_components=components, compose=compose,
        ref_point=ref_point, ideal=ideal,
    )


if __name__ == "__main__":
    torch.set_default_dtype(DTYPE)
    X = SobolEngine(6, scramble=True, seed=5).draw(2000).to(DTYPE)
    print(f"{'g1':9s} {'injective':>9s} {'f1 range':>22s} {'f2 range':>22s} "
          f"{'ref point':>18s}")
    for name in OUTER_G1:
        p = make_two_stage(name)
        Y = p.evaluate(X)
        print(f"{name:9s} {str(G1_INJECTIVE[name]):>9s} "
              f"[{Y[:,0].min():+7.3f},{Y[:,0].max():+7.3f}] "
              f"[{Y[:,1].min():+7.3f},{Y[:,1].max():+7.3f}] "
              f"[{p.ref_point[0]:+6.2f},{p.ref_point[1]:+6.2f}]")
    print("\nf2 is identical across the three rows by construction "
          "(same h1..h4, same Ackley).")
