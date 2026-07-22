r"""Direct-vs-composite BO solvers, vendored from a collaborator's repo.

Source: https://github.com/tau315/composite-mobo (`solvers.py`), by tau315.
Vendored here (with attribution, not reimplemented) because this project
needs to run these specific solvers -- not the repo's own MORBO variants
(`morbo`/`composite_morbo`/`batched_morbo`/`composite_batched_morbo`), since
this project already has its OWN authoritative MORBO implementation
(`morbo/`) and its own direct-vs-composite comparison for that engine
(`morbo/problems/composite_dtlz2*.py`, `plug_and_play/benchmarks.py`'s
``composite_dtlz2``). What this project does NOT already have is a
direct-vs-composite comparison for these non-MORBO solvers -- that is
the entire point of `composite_ablation/`.

Trimmed to exactly the six solvers this project runs (three direct/
composite pairs) plus their shared helpers:

    standard_mobo               / composite_mobo
    chebyshev_bo                / composite_chebyshev_bo
    spherical_chebyshev_bo      / composite_spherical_chebyshev_bo

Everything here keeps the source repo's own conventions unchanged so it
stays a faithful transcription, not a reinterpretation:
MINIMIZATION on ``[0, 1]^d``; composite solvers take
``evaluate_components(X) -> H`` and ``compose(H) -> Y`` with
``compose(evaluate_components(X)) == evaluate(X)``. See
``composite_ablation/adapters.py`` for the bridge from this project's own
MAXIMIZATION-convention ``Benchmark`` objects (``plug_and_play/benchmarks.py``)
into this convention.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import shutil
import sys
from typing import Callable, Optional, Sequence

import torch
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.acquisition.multi_objective.logei import (
    qLogExpectedHypervolumeImprovement,
)
import botorch.acquisition.multi_objective.logei as _multi_objective_logei
from botorch.acquisition.objective import GenericMCObjective
from botorch.fit import fit_gpytorch_mll
try:
    # Only present in newer BoTorch than this project pins
    # (botorch>=0.6, currently installed: 0.9.5); fall back to the
    # broader RuntimeError optimize_acqf raises for the same failure
    # mode on older versions so this stays a functional no-op elsewhere.
    from botorch.exceptions.errors import OptimizationGradientError
except ImportError:
    OptimizationGradientError = RuntimeError
from botorch.models import ModelListGP, SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    NondominatedPartitioning,
)
from gpytorch.mlls import ExactMarginalLogLikelihood, SumMarginalLogLikelihood
from gpytorch.constraints import GreaterThan
from gpytorch.kernels import Kernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.priors import LogNormalPrior

Tensor = torch.Tensor
Evaluator = Callable[[Tensor], Tensor]
Composer = Callable[[Tensor], Tensor]

# Same Windows/no-MSVC workaround the source file uses -- BoTorch tries to
# JIT-build an optional fused qLogEHVI kernel; without a C compiler this
# produces a long subprocess traceback before falling back to the identical
# pure-Python path anyway. Preempt it.
if sys.platform == "win32" and shutil.which("cl") is None:
    _multi_objective_logei._load_attempted = True


@dataclass
class SolverResult:
    """Evaluated design and objective data returned by every solver."""

    X: Tensor
    Y: Tensor
    components: Optional[Tensor] = None
    weights: Optional[Tensor] = None
    run_ids: Optional[Tensor] = None


def smooth_tchebycheff(
    Y: Tensor, weights: Tensor, ideal: Tensor, temperature: float = 0.05
) -> Tensor:
    """Smooth Tchebycheff loss for minimization: `t log sum_i exp(w_i (y_i-z_i*) / t)`."""

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    return temperature * torch.logsumexp(weights * (Y - ideal) / temperature, dim=-1)


def simplex_weights(n: int, m: int, *, seed: int = 0) -> Tensor:
    """Deterministic boundary-covering weights for 2-D, Dirichlet otherwise."""

    if n < 1 or m < 2:
        raise ValueError("n >= 1 and m >= 2 are required")
    if m == 2:
        w = torch.linspace(0.05, 0.95, n, dtype=torch.double)
        return torch.stack((w, 1.0 - w), dim=-1)
    generator = torch.Generator().manual_seed(seed)
    e = -torch.log(torch.rand(n, m, generator=generator, dtype=torch.double))
    return e / e.sum(dim=-1, keepdim=True)


def _sobol(n: int, d: int, seed: int) -> Tensor:
    return torch.quasirandom.SobolEngine(d, scramble=True, seed=seed).draw(n).double()


def _independent_gp(X: Tensor, Y: Tensor) -> ModelListGP:
    models = [
        SingleTaskGP(X, Y[:, i : i + 1], outcome_transform=Standardize(m=1))
        for i in range(Y.shape[-1])
    ]
    model = ModelListGP(*models)
    fit_gpytorch_mll(SumMarginalLogLikelihood(model.likelihood, model))
    return model


def _optimize(acq, d: int, raw_samples: int, num_restarts: int) -> Tensor:
    bounds = torch.stack((torch.zeros(d), torch.ones(d))).double()
    try:
        X, _ = optimize_acqf(
            acq,
            bounds=bounds,
            q=1,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            options={"batch_limit": 5, "maxiter": 200},
        )
        return X.detach()
    except OptimizationGradientError:
        # A nonlinear composite map can occasionally make a local acquisition
        # gradient non-finite. Preserve the BO run by selecting the best finite
        # acquisition value from a fresh, space-filling Sobol candidate set.
        candidates = draw_sobol_samples(
            bounds=bounds, n=max(raw_samples * num_restarts, 256), q=1
        )
        with torch.no_grad():
            values = acq(candidates)
        values = torch.nan_to_num(values, nan=-torch.inf, neginf=-torch.inf)
        if not torch.isfinite(values).any():
            raise RuntimeError("acquisition was non-finite on every fallback candidate")
        return candidates[values.argmax()].detach()


def _check_composition(C: Tensor, Y: Tensor, compose: Composer) -> None:
    reconstructed = compose(C)
    if reconstructed.shape != Y.shape or not torch.allclose(
        reconstructed, Y, atol=1e-7, rtol=1e-5
    ):
        raise ValueError("compose(evaluate_components(X)) must equal evaluate(X)")


def standard_mobo(
    evaluate: Evaluator,
    dim: int,
    ref_point: Tensor,
    *,
    n_init: int = 5,
    n_iter: int = 40,
    seed: int = 0,
    raw_samples: int = 128,
    num_restarts: int = 8,
) -> SolverResult:
    """Independent objective GPs followed by numerically stable sequential qLogEHVI."""

    torch.manual_seed(seed)
    X = _sobol(n_init, dim, seed)
    Y = evaluate(X).double()
    ref_max = -torch.as_tensor(ref_point, dtype=torch.double)
    for _ in range(n_iter):
        model = _independent_gp(X, -Y)
        partitioning = NondominatedPartitioning(ref_point=ref_max, Y=-Y)
        acq = qLogExpectedHypervolumeImprovement(
            model=model, ref_point=ref_max.tolist(), partitioning=partitioning
        )
        x = _optimize(acq, dim, raw_samples, num_restarts)
        X, Y = torch.cat((X, x)), torch.cat((Y, evaluate(x).double()))
    return SolverResult(X=X, Y=Y)


def composite_mobo(
    evaluate: Evaluator,
    evaluate_components: Evaluator,
    compose: Composer,
    dim: int,
    ref_point: Tensor,
    *,
    n_init: int = 5,
    n_iter: int = 40,
    seed: int = 0,
    raw_samples: int = 128,
    num_restarts: int = 8,
) -> SolverResult:
    """Intermediate-node GPs and qLogEHVI on composed samples (MO-BOCF)."""

    from botorch.acquisition.multi_objective.objective import (
        GenericMCMultiOutputObjective,
    )

    torch.manual_seed(seed)
    X = _sobol(n_init, dim, seed)
    C, Y = evaluate_components(X).double(), evaluate(X).double()
    _check_composition(C, Y, compose)
    ref_max = -torch.as_tensor(ref_point, dtype=torch.double)
    objective = GenericMCMultiOutputObjective(lambda samples, X=None: -compose(samples))
    for _ in range(n_iter):
        model = _independent_gp(X, C)
        partitioning = NondominatedPartitioning(ref_point=ref_max, Y=-Y)
        acq = qLogExpectedHypervolumeImprovement(
            model=model,
            ref_point=ref_max.tolist(),
            partitioning=partitioning,
            objective=objective,
        )
        x = _optimize(acq, dim, raw_samples, num_restarts)
        c, y = evaluate_components(x).double(), evaluate(x).double()
        X, C, Y = torch.cat((X, x)), torch.cat((C, c)), torch.cat((Y, y))
    return SolverResult(X=X, Y=Y, components=C)


def _scalarized_runs(
    evaluate: Evaluator,
    dim: int,
    weights: Tensor,
    ideal: Tensor,
    temperature: float,
    n_init: int,
    n_per_scalarization: int,
    seed: int,
    raw_samples: int,
    num_restarts: int,
    evaluate_components: Optional[Evaluator] = None,
    compose: Optional[Composer] = None,
) -> SolverResult:
    """Branch every weight from one shared, once-evaluated initial design."""
    torch.manual_seed(seed)
    X_initial = _sobol(n_init, dim, seed)
    Y_initial = evaluate(X_initial).double()
    C_initial = (
        evaluate_components(X_initial).double()
        if evaluate_components is not None
        else None
    )
    if C_initial is not None:
        _check_composition(C_initial, Y_initial, compose)  # type: ignore[arg-type]

    all_x, all_y = [X_initial], [Y_initial]
    all_c = [C_initial] if C_initial is not None else []
    ids = [torch.full((n_init,), -1, dtype=torch.long)]
    for weight_id, weight in enumerate(weights.double()):
        torch.manual_seed(seed + 104729 * weight_id)
        X, Y = X_initial.clone(), Y_initial.clone()
        C = C_initial.clone() if C_initial is not None else None
        for _ in range(n_per_scalarization):
            if C is None:
                model = _independent_gp(X, Y)
                objective = GenericMCObjective(
                    lambda samples, X=None, w=weight: (
                        -smooth_tchebycheff(samples, w, ideal, temperature)
                    )
                )
            else:
                model = _independent_gp(X, C)
                objective = GenericMCObjective(
                    lambda samples, X=None, w=weight: (
                        -smooth_tchebycheff(
                            compose(samples),
                            w,
                            ideal,
                            temperature,  # type: ignore[misc]
                        )
                    )
                )
            observed_utility = -smooth_tchebycheff(Y, weight, ideal, temperature)
            acq = qLogExpectedImprovement(
                model=model, best_f=observed_utility.max(), objective=objective
            )
            x = _optimize(acq, dim, raw_samples, num_restarts)
            y = evaluate(x).double()
            X, Y = torch.cat((X, x)), torch.cat((Y, y))
            if C is not None:
                C = torch.cat((C, evaluate_components(x).double()))  # type: ignore[misc]
        all_x.append(X[n_init:])
        all_y.append(Y[n_init:])
        ids.append(torch.full((n_per_scalarization,), weight_id, dtype=torch.long))
        if C is not None:
            all_c.append(C[n_init:])
    return SolverResult(
        X=torch.cat(all_x),
        Y=torch.cat(all_y),
        components=torch.cat(all_c) if all_c else None,
        weights=weights,
        run_ids=torch.cat(ids),
    )


def chebyshev_bo(
    evaluate: Evaluator,
    dim: int,
    weights: Tensor,
    ideal: Tensor,
    *,
    temperature: float = 0.05,
    n_init: int = 5,
    n_per_scalarization: int = 5,
    seed: int = 0,
    raw_samples: int = 128,
    num_restarts: int = 8,
) -> SolverResult:
    """Objective GPs and EI through STCH posterior samples, one run per weight."""

    return _scalarized_runs(
        evaluate,
        dim,
        weights,
        ideal.double(),
        temperature,
        n_init,
        n_per_scalarization,
        seed,
        raw_samples,
        num_restarts,
    )


def composite_chebyshev_bo(
    evaluate: Evaluator,
    evaluate_components: Evaluator,
    compose: Composer,
    dim: int,
    weights: Tensor,
    ideal: Tensor,
    *,
    temperature: float = 0.05,
    n_init: int = 5,
    n_per_scalarization: int = 5,
    seed: int = 0,
    raw_samples: int = 128,
    num_restarts: int = 8,
) -> SolverResult:
    """Node GPs and EI on the double-composed smooth-Tchebycheff posterior."""

    return _scalarized_runs(
        evaluate,
        dim,
        weights,
        ideal.double(),
        temperature,
        n_init,
        n_per_scalarization,
        seed,
        raw_samples,
        num_restarts,
        evaluate_components,
        compose,
    )


# ---------------------------------------------------------------------------
# High-dimensional spherical-linear BO (Doumont et al., AISTATS 2026)
# ---------------------------------------------------------------------------


def inverse_stereographic_projection(X: Tensor) -> Tensor:
    """Map R^d bijectively to the unit sphere S^d (paper Eq. 4)."""

    norm2 = X.square().sum(dim=-1, keepdim=True)
    return torch.cat((2.0 * X, norm2 - 1.0), dim=-1) / (1.0 + norm2)


class SphericalLinearKernel(Kernel):
    """Paper-faithful spherical linear kernel.

    Inputs are centered, divided by ARD scales and a decoupled global scale,
    inverse-stereographically projected, and evaluated with
    k(x,x') = b0 + b1 P(z)^T P(z'), where (b0,b1) lies on the simplex.
    """

    has_lengthscale = True

    def __init__(self, dim: int, bounds: tuple[float, float] = (0.0, 1.0)):
        prior = LogNormalPrior(math.sqrt(2.0), math.sqrt(3.0))
        super().__init__(
            ard_num_dims=dim,
            lengthscale_prior=prior,
            lengthscale_constraint=GreaterThan(
                2.5e-2, transform=None, initial_value=prior.mode
            ),
        )
        self.register_buffer("centers", torch.full((dim,), sum(bounds) / 2))
        self.register_buffer("widths", torch.full((dim,), bounds[1] - bounds[0]))
        self.register_parameter("raw_coeffs", torch.nn.Parameter(torch.zeros(2)))
        self.register_parameter("raw_global_scale", torch.nn.Parameter(torch.zeros(1)))

    @property
    def coefficients(self) -> Tensor:
        return torch.softmax(self.raw_coeffs, dim=-1)

    def _features(self, X: Tensor) -> Tensor:
        scaled = (X - self.centers) / self.lengthscale
        max_norm2 = (
            (self.widths / (2.0 * self.lengthscale.squeeze(-2))).square().sum(-1)
        )
        global_scale = torch.sqrt(torch.sigmoid(self.raw_global_scale) * max_norm2)
        projected = inverse_stereographic_projection(scaled / global_scale)
        b0, b1 = self.coefficients
        return torch.cat(
            (projected * b1.sqrt(), b0.sqrt().expand(*projected.shape[:-1], 1)),
            dim=-1,
        )

    def forward(self, x1: Tensor, x2: Tensor, diag: bool = False, **params):
        phi1, phi2 = self._features(x1), self._features(x2)
        if diag:
            return (phi1 * phi2).sum(dim=-1)
        return phi1 @ phi2.transpose(-1, -2)


def _spherical_gp(X: Tensor, Y: Tensor) -> ModelListGP:
    """Independent paper-style spherical-linear GPs for every output."""

    models = []
    for i in range(Y.shape[-1]):
        noise_prior = LogNormalPrior(-4.0, 1.0)
        likelihood = GaussianLikelihood(
            noise_prior=noise_prior,
            noise_constraint=GreaterThan(1e-4, initial_value=noise_prior.mode),
        )
        models.append(
            SingleTaskGP(
                X,
                Y[:, i : i + 1],
                mean_module=ConstantMean(),
                covar_module=SphericalLinearKernel(X.shape[-1]),
                likelihood=likelihood,
                outcome_transform=Standardize(m=1),
            )
        )
    model = ModelListGP(*models)
    fit_gpytorch_mll(SumMarginalLogLikelihood(model.likelihood, model))
    return model


def _high_dim_scalarized_runs(
    evaluate: Evaluator,
    dim: int,
    weights: Tensor,
    ideal: Tensor,
    *,
    temperature: float = 0.05,
    n_init: int = 5,
    n_per_scalarization: int = 5,
    seed: int = 0,
    raw_samples: int = 256,
    num_restarts: int = 10,
    evaluate_components: Optional[Evaluator] = None,
    compose: Optional[Composer] = None,
) -> SolverResult:
    """Spherical-linear STCH with one shared initial design."""
    torch.manual_seed(seed)
    X_initial = _sobol(n_init, dim, seed)
    Y_initial = evaluate(X_initial).double()
    C_initial = (
        evaluate_components(X_initial).double()
        if evaluate_components is not None
        else None
    )
    if C_initial is not None:
        _check_composition(C_initial, Y_initial, compose)  # type: ignore[arg-type]

    all_x, all_y = [X_initial], [Y_initial]
    all_c = [C_initial] if C_initial is not None else []
    all_ids = [torch.full((n_init,), -1, dtype=torch.long)]
    for weight_id, weight in enumerate(weights.double()):
        torch.manual_seed(seed + 104729 * weight_id)
        X, Y = X_initial.clone(), Y_initial.clone()
        C = C_initial.clone() if C_initial is not None else None
        for _ in range(n_per_scalarization):
            training_values = Y if C is None else C
            model = _spherical_gp(X, training_values)
            if C is None:
                objective = GenericMCObjective(
                    lambda samples, X=None, w=weight: (
                        -smooth_tchebycheff(samples, w, ideal, temperature)
                    )
                )
            else:
                objective = GenericMCObjective(
                    lambda samples, X=None, w=weight: (
                        -smooth_tchebycheff(
                            compose(samples),
                            w,
                            ideal,
                            temperature,  # type: ignore[misc]
                        )
                    )
                )
            observed_utility = -smooth_tchebycheff(Y, weight, ideal, temperature)
            acq = qLogExpectedImprovement(
                model=model, best_f=observed_utility.max(), objective=objective
            )
            x = _optimize(acq, dim, raw_samples, num_restarts)
            y = evaluate(x).double()
            X, Y = torch.cat((X, x)), torch.cat((Y, y))
            if C is not None:
                C = torch.cat((C, evaluate_components(x).double()))  # type: ignore[misc]
        all_x.append(X[n_init:])
        all_y.append(Y[n_init:])
        all_ids.append(torch.full((n_per_scalarization,), weight_id, dtype=torch.long))
        if C is not None:
            all_c.append(C[n_init:])
    return SolverResult(
        X=torch.cat(all_x),
        Y=torch.cat(all_y),
        components=torch.cat(all_c) if all_c else None,
        weights=weights,
        run_ids=torch.cat(all_ids),
    )


def spherical_chebyshev_bo(
    evaluate: Evaluator, dim: int, weights: Tensor, ideal: Tensor, **kwargs
) -> SolverResult:
    """Eight-weight STCH using five acquisitions per weight by default."""
    return _high_dim_scalarized_runs(evaluate, dim, weights, ideal.double(), **kwargs)


def composite_spherical_chebyshev_bo(
    evaluate: Evaluator,
    evaluate_components: Evaluator,
    compose: Composer,
    dim: int,
    weights: Tensor,
    ideal: Tensor,
    **kwargs,
) -> SolverResult:
    """Spherical-linear GPs on h_ij, followed by composition and STCH."""
    return _high_dim_scalarized_runs(
        evaluate,
        dim,
        weights,
        ideal.double(),
        evaluate_components=evaluate_components,
        compose=compose,
        **kwargs,
    )
