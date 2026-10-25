"""Where does the two-objective composite advantage come from?  Regression only.

The hypervolume experiment says Composite gains most on g1 = z^3 and less on
z^2, which is backwards if non-injectivity is what helps.  Hypervolume is a
single noisy endpoint, so this asks the same question by plain regression,
objective by objective, with no optimiser anywhere.

    Direct     a GP per objective, fitted to f1 and f2
    Composite  a GP per intermediate, fitted to h1 and h2, then push through g

Because g2 is the identity, f2 = h2 and the two models are doing exactly the
same thing on the second objective.  Any difference between the arms has to
appear on f1 alone, and the per-objective split shows whether it does.

    python composite_ablation/two_stage_rmse.py
"""
from __future__ import annotations

import argparse
import math
import os
import statistics as st
import sys
import warnings

import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch.quasirandom import SobolEngine

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from composite_ablation.two_stage import OUTER_G1, make_two_stage            # noqa: E402

warnings.filterwarnings("ignore")
DTYPE = torch.double
N_SAMPLES = 2048


def _gp(X, y):
    m = SingleTaskGP(X, y.unsqueeze(-1), outcome_transform=Standardize(m=1))
    try:
        fit_gpytorch_mll(ExactMarginalLogLikelihood(m.likelihood, m))
    except Exception:
        m.eval()
    return m


def nrmse(pred, truth):
    """RMSE divided by the spread of the target: 1.0 = no better than the mean."""
    return float(((pred - truth) ** 2).mean().sqrt() / truth.std())


def one_split(p, n_train, n_test, seed):
    X = SobolEngine(p.dim, scramble=True, seed=seed).draw(n_train + n_test).to(DTYPE)
    Xtr, Xte = X[:n_train], X[n_train:]
    Ctr, Cte = p.evaluate_components(Xtr), p.evaluate_components(Xte)
    Ytr, Yte = p.evaluate(Xtr), p.evaluate(Xte)

    out = {}
    for i in (0, 1):
        gp = _gp(Xtr, Ytr[:, i])
        with torch.no_grad():
            out[f"direct_f{i+1}"] = nrmse(gp.posterior(Xte).mean.squeeze(-1), Yte[:, i])

    gen = torch.Generator().manual_seed(seed * 7 + 3)
    mus, sds = [], []
    for j in range(Ctr.shape[-1]):
        gp = _gp(Xtr, Ctr[:, j])
        with torch.no_grad():
            post = gp.posterior(Xte)
        mu = post.mean.squeeze(-1)
        mus.append(mu)
        sds.append(post.variance.clamp_min(1e-12).sqrt().squeeze(-1))
        out[f"comp_h{j+1}"] = nrmse(mu, Cte[:, j])
    eps = torch.randn(N_SAMPLES, len(mus), n_test, generator=gen, dtype=DTYPE)
    draws = torch.stack([m_ + s_ * eps[:, k] for k, (m_, s_) in enumerate(zip(mus, sds))],
                        dim=-1)
    pred = p.compose(draws).mean(0)                        # (n_test, 2)
    for i in (0, 1):
        out[f"comp_f{i+1}"] = nrmse(pred[:, i], Yte[:, i])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--g1", nargs="+", default=list(OUTER_G1))
    ap.add_argument("--g2", default="identity", choices=["identity", "ackley"])
    ap.add_argument("--kernel", default="matern", choices=["matern", "rbf"])
    ap.add_argument("--n-train", type=int, default=80)
    ap.add_argument("--n-test", type=int, default=20)
    ap.add_argument("--splits", type=int, default=10)
    ap.add_argument("--draws", type=int, default=3)
    a = ap.parse_args()
    torch.set_default_dtype(DTYPE)

    print(f"g2 = {a.g2}, h drawn from {a.kernel}, {a.n_train} train / "
          f"{a.n_test} test, {a.draws * a.splits} fits per condition")
    print("normalised RMSE: 1.0 means no better than predicting the mean\n")
    print(f"{'g1':9s} | {'OBJECTIVE 1 (varies)':>30s} | {'OBJ 2 (identity)':>21s} | "
          f"{'inner':>15s}")
    print(f"{'':9s} | {'direct':>9s} {'composite':>11s} {'gain':>8s} | "
          f"{'direct':>9s} {'composite':>11s} | {'h1':>7s} {'h2':>7s}")
    print("-" * 92)
    for name in a.g1:
        acc = {}
        for draw in range(a.draws):
            p = make_two_stage(name, draw_seed=draw, g2=a.g2, kernel=a.kernel)
            for s in range(a.splits):
                r = one_split(p, a.n_train, a.n_test, 1000 * draw + s)
                for k, v in r.items():
                    acc.setdefault(k, []).append(v)
        m = {k: st.mean(v) for k, v in acc.items()}
        gain = m["direct_f1"] - m["comp_f1"]
        print(f"{name:9s} | {m['direct_f1']:9.3f} {m['comp_f1']:11.3f} {gain:+8.3f} | "
              f"{m['direct_f2']:9.3f} {m['comp_f2']:11.3f} | "
              f"{m['comp_h1']:7.3f} {m['comp_h2']:7.3f}", flush=True)
    print("\nObjective 2 is the control: f2 = h2, so both arms fit the same GP to the")
    print("same data and their columns should agree.  The inner columns should be")
    print("identical across rows, since h1 and h2 never change.")


if __name__ == "__main__":
    main()
