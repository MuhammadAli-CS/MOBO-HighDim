"""Random Fourier feature draws, vendored from the three-arm-composite repo.

    SOURCE OF TRUTH:  three-arm-composite/three_arm/test_functions.py
    Do not edit here.  Re-copy _spectral_sample and RFFDraw from there.

This file exists because composite_ablation/two_stage.py needs exactly these
two objects and nothing else from that module, and the alternative -- keeping
a whole stale copy of the `three_arm` package in this repo -- actively caused
a bug.  That copy was an older fork whose Matern sampler drew the frequency
scale with torch.distributions.Gamma(...).sample(), which takes no generator
and reads the global RNG, so every two-objective Matern problem was a
different random function on each construction.  Because the fork looked like
the real package, fixes made in the other repo silently did not apply here.

Keeping the surface down to two objects makes that divergence easy to spot:

    python -c "import hashlib,re,sys;         a=open('../three-arm-composite/three_arm/test_functions.py').read();         b=open('composite_ablation/rff.py').read();         f=lambda s,k: re.search(k+r'.*?(?=
(?:class|def|# ---)\s)', s, re.S).group();         print(f(a,'class RFFDraw')==f(b,'class RFFDraw'))"
"""
from __future__ import annotations

import math

import torch
from torch.quasirandom import SobolEngine

DTYPE = torch.double
TWO_PI = 2.0 * math.pi
_CHUNK = 4096


def _spectral_sample(d, n_features, lengthscale, kernel, gen):
    """Draw frequencies from the kernel's spectral density.

    RBF's is Gaussian.  Matern-nu's is a multivariate Student-t with 2*nu
    degrees of freedom, sampled as a Gaussian scaled by an inverse Gamma --
    which is what makes Matern draws rougher than RBF ones.  BoTorch's default
    kernel is Matern 5/2, so generating h from it is what makes the surrogate
    correctly specified about the inner function rather than merely close.
    """
    z = torch.randn(n_features, d, generator=gen, dtype=DTYPE)
    if kernel == "rbf":
        return z / lengthscale
    # u ~ Gamma(nu, nu), so omega = z / (lengthscale * sqrt(u)) is Student-t
    # with 2*nu degrees of freedom.  torch.distributions.Gamma.sample() takes
    # no generator and reads the global RNG, so using it made every Matern
    # problem a DIFFERENT random function on each construction -- the draw
    # seed stopped selecting anything, maps no longer shared an h, and a
    # stored f_min could not be recomputed afterwards.
    #
    # Sampling it from the seeded generator instead, via the decomposition
    #     Gamma(2.5, 1) = Exp(1) + Exp(1) + Gamma(0.5, 1),
    # with Exp(1) = -log U and Gamma(0.5, 1) = Z^2 / 2.  Exact, and only needs
    # rand/randn, both of which accept the generator.
    nu = 2.5
    u1 = torch.rand(n_features, 1, generator=gen, dtype=DTYPE).clamp_min(1e-300)
    u2 = torch.rand(n_features, 1, generator=gen, dtype=DTYPE).clamp_min(1e-300)
    zh = torch.randn(n_features, 1, generator=gen, dtype=DTYPE)
    u = (-u1.log() - u2.log() + 0.5 * zh.square()) / nu
    return z / (lengthscale * u.sqrt())


class RFFDraw:
    """Fixed RBF-GP draw via random Fourier features.

    h(x) = sqrt(2/M) sum_m a_m cos(w_m^T x + b_m) with a ~ N(0,1),
    b ~ U(0, 2pi), w ~ N(0, l^{-2} I), whose prior is GP(0, k_RBF) with unit
    outputscale.  The draw is standardized on a Sobol reference set as the doc
    specifies, so the implied prior on the standardized function is
    GP(-mu/sd, k_RBF / sd^2).  That prior is exposed as ``prior_mean`` and
    ``prior_var`` and is the known prior shared by the Composite and Latent arms.
    """

    def __init__(self, d: int, lengthscale: float = 0.5, seed: int = 0,
                 n_features: int = 1024, n_ref: int = 16384,
                 kernel: str = "rbf"):
        gen = torch.Generator().manual_seed(seed)
        self.kernel = kernel
        self.d, self.lengthscale, self.seed = d, lengthscale, seed
        self.omega = _spectral_sample(d, n_features, lengthscale, kernel, gen)
        self.b = TWO_PI * torch.rand(n_features, generator=gen, dtype=DTYPE)
        self.a = torch.randn(n_features, generator=gen, dtype=DTYPE)
        self.scale = math.sqrt(2.0 / n_features)
        ref = SobolEngine(d, scramble=True, seed=7919 + seed).draw(n_ref).to(DTYPE)
        raw = self._raw(ref)
        self.mu, self.sd = raw.mean().item(), raw.std().item()

    def _raw(self, X: torch.Tensor) -> torch.Tensor:
        if X.shape[0] <= _CHUNK:
            return self.scale * (torch.cos(X @ self.omega.T + self.b) @ self.a)
        return torch.cat([self._raw(X[i:i + _CHUNK]) for i in range(0, X.shape[0], _CHUNK)])

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        return (self._raw(X) - self.mu) / self.sd

    @property
    def prior_mean(self) -> float:
        return -self.mu / self.sd

    @property
    def prior_var(self) -> float:
        return 1.0 / self.sd ** 2
