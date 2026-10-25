"""Composite vs Direct on the two-objective structure, varying only g1.

    python composite_ablation/run_two_stage.py --trials 10 --draws 4

Reports the final-hypervolume gain of Composite over Direct for each choice of
g1, paired over seeds.  Everything but g1 is held fixed -- input dimension,
the five inner functions, g2 (Ackley), the reference point and the initial
designs -- so differences between rows are attributable to g1's structure.

Expected from the single-objective study:

    identity   g1 linear          direct model is correct; smallest gain
    cube       bijective, nonlinear   wrong model shape, nothing lost
    square     not bijective      wrong model shape AND information lost

so identity -> cube isolates misspecification and cube -> square adds the
information channel at comparable nonlinearity.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics as st
import sys
import time
import warnings
from pathlib import Path

import torch
from botorch.utils.multi_objective.hypervolume import Hypervolume

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from composite_ablation.solvers import composite_mobo, standard_mobo       # noqa: E402
from composite_ablation.two_stage import (                                 # noqa: E402
    G1_INJECTIVE, OUTER_G1, make_two_stage,
)

warnings.filterwarnings("ignore")
DTYPE = torch.double


def final_hv(Y: torch.Tensor, ref_point: torch.Tensor) -> float:
    """Hypervolume of the observed front, in the solvers' minimise convention."""
    hv = Hypervolume(ref_point=-ref_point.to(DTYPE))
    return float(hv.compute(-Y.to(DTYPE)))


def hv_trace(Y: torch.Tensor, ref_point: torch.Tensor, n_init: int):
    """Hypervolume after each adaptive evaluation, so improvement can be
    plotted rather than only its endpoint."""
    hv = Hypervolume(ref_point=-ref_point.to(DTYPE))
    Yd = -Y.to(DTYPE)
    return [float(hv.compute(Yd[: n_init + t])) for t in range(Y.shape[0] - n_init + 1)]


def _unit(args):
    """One (g1, draw, seed): both arms on the same problem and initial design.

    Parallelising at this granularity rather than per condition keeps every
    worker busy -- there are only three conditions but dozens of runs.  No
    max_tasks_per_child: when every worker hits the recycle limit on the same
    tick, ProcessPoolExecutor.map deadlocks on Windows.
    """
    g1_name, draw, seed, dim, n_init, n_iter, g2, kernel = args
    torch.set_num_threads(1)
    torch.set_default_dtype(DTYPE)
    warnings.filterwarnings("ignore")
    p = make_two_stage(g1_name, dim=dim, draw_seed=draw, g2=g2, kernel=kernel)
    rd = standard_mobo(p.evaluate, p.dim, p.ref_point,
                       n_init=n_init, n_iter=n_iter, seed=seed)
    rc = composite_mobo(p.evaluate, p.evaluate_components, p.compose,
                        p.dim, p.ref_point, n_init=n_init, n_iter=n_iter, seed=seed)
    return (g1_name,
            hv_trace(rd.Y, p.ref_point, n_init),
            hv_trace(rc.Y, p.ref_point, n_init))


def run_all(names, trials, draws, dim, n_init, n_iter, jobs, g2, kernel):
    units = [(n, d, s, dim, n_init, n_iter, g2, kernel)
             for n in names for d in range(draws) for s in range(trials)]
    out = {n: ([], []) for n in names}
    if jobs > 1:
        import concurrent.futures as cf
        done = 0
        with cf.ProcessPoolExecutor(max_workers=jobs) as ex:
            for name, dv, cv in ex.map(_unit, units):
                out[name][0].append(dv); out[name][1].append(cv)
                done += 1
                print(f"  {done}/{len(units)} runs", end="\r", flush=True)
        print(" " * 30, end="\r")
    else:
        for u in units:
            name, dv, cv = _unit(u)
            out[name][0].append(dv); out[name][1].append(cv)
    return out


def paired_pct(direct, comp):
    """Per-run percentage gain, then mean and standard error over runs."""
    pct = [100.0 * (c - d) / d if d > 0 else float("nan")
           for d, c in zip(direct, comp)]
    pct = [v for v in pct if v == v]
    m = st.mean(pct)
    se = st.stdev(pct) / math.sqrt(len(pct)) if len(pct) > 1 else float("nan")
    return m, se, sum(1 for c, d in zip(comp, direct) if c > d), len(pct)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials", type=int, default=10, help="seeds per function draw")
    ap.add_argument("--draws", type=int, default=4, help="independent inner functions")
    ap.add_argument("--dim", type=int, default=6)
    ap.add_argument("--n-init", type=int, default=5)
    ap.add_argument("--n-iter", type=int, default=40)
    ap.add_argument("--g1", nargs="+", default=list(OUTER_G1))
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--kernel", choices=["rbf", "matern"], default="matern")
    ap.add_argument("--g2", choices=["ackley", "identity"], default="ackley",
                    help="identity makes objective 2 a plain GP, so the"
                         " whole advantage is attributable to g1")
    ap.add_argument("--out", default="composite_ablation/results/two_stage.json")
    a = ap.parse_args()
    torch.set_default_dtype(DTYPE)

    print(f"dim={a.dim}, n_init={a.n_init}, n_iter={a.n_iter}, "
          f"{a.draws} draws x {a.trials} seeds = {a.draws * a.trials} runs per arm\n")
    print(f"{'g1':9s} {'injective':>9s} {'HV direct':>11s} {'HV composite':>13s} "
          f"{'gain %':>17s} {'wins':>8s} {'min':>6s}")
    print("-" * 82)
    t0 = time.time()
    res = run_all(a.g1, a.trials, a.draws, a.dim, a.n_init, a.n_iter, a.jobs, a.g2, a.kernel)
    blob = {}
    for name in a.g1:
        dt, ct = res[name]
        d = [t[-1] for t in dt]; c = [t[-1] for t in ct]
        m, se, wins, n = paired_pct(d, c)
        blob[name] = dict(direct=d, composite=c, injective=G1_INJECTIVE[name],
                          direct_trace=dt, composite_trace=ct)
        print(f"{name:9s} {str(G1_INJECTIVE[name]):>9s} {st.mean(d):11.4f} "
              f"{st.mean(c):13.4f} {m:+9.2f} +/- {se:5.2f} {wins:5d}/{n} "
              f"{(time.time()-t0)/60:5.1f}", flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"config": vars(a), "results": blob}, open(a.out, "w"))
    print(f"\nwrote {a.out}")
    print("gain is the per-run percentage difference in final hypervolume, "
          "paired over the shared initial design.")


if __name__ == "__main__":
    main()
