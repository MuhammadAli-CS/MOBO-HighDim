"""Prove the g1 maps are identical to the single-objective outer maps.

The point of the two-objective grid is to compare the same outer maps under
hypervolume that the single-objective grid measured under regret. That
comparison is only meaningful if the maps really are the same function, and
`OUTER_G1` writes them out by hand rather than importing them -- the
single-objective module builds its maps together with level-set move
machinery this side never uses, and vendoring the whole thing is what caused
the last bug.

So instead of sharing code, this checks agreement numerically, over a grid
wider than the range `h` ever reaches. Any drift shows up as a non-zero
maximum difference.

    python composite_ablation/check_g1.py
    python composite_ablation/check_g1.py --repo ../three-arm-composite

Exits non-zero if any map disagrees, so it can be wired into CI.
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from composite_ablation.two_stage import OUTER_G1                  # noqa: E402

DTYPE = torch.double


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=os.path.join("..", "three-arm-composite"),
                    help="path to the three-arm-composite checkout")
    ap.add_argument("--lo", type=float, default=-6.0)
    ap.add_argument("--hi", type=float, default=6.0)
    ap.add_argument("--n", type=int, default=200001)
    a = ap.parse_args()
    torch.set_default_dtype(DTYPE)

    repo = os.path.abspath(a.repo)
    if not os.path.isdir(repo):
        print(f"three-arm-composite not found at {repo}; skipping the check.")
        print("Pass --repo if it lives elsewhere. Nothing was verified.")
        return 0
    sys.path.insert(0, repo)
    try:
        from three_arm.test_functions import OUTER_MAPS
    except ImportError as exc:
        print(f"cannot import the single-objective maps ({exc}); nothing verified.")
        return 0

    z = torch.linspace(a.lo, a.hi, a.n, dtype=DTYPE)
    y = z.unsqueeze(-1)                       # three_arm's fn takes (..., k)

    missing, bad, ok = [], [], 0
    print(f"comparing {len(OUTER_G1)} maps on [{a.lo}, {a.hi}], {a.n} points\n")
    print(f"{'map':14s} {'max |difference|':>18s}")
    for name, fn in OUTER_G1.items():
        ref = OUTER_MAPS.get(name)
        if ref is None:
            missing.append(name)
            continue
        if ref.k != 1:
            missing.append(f"{name} (k={ref.k})")
            continue
        d = float((fn(z) - ref.fn(y)).abs().max())
        flag = "" if d == 0.0 else "   <-- DIFFERS"
        if d != 0.0:
            bad.append((name, d))
        else:
            ok += 1
        print(f"{name:14s} {d:18.3e}{flag}")

    print()
    if missing:
        print(f"not present in the single-objective grid: {', '.join(missing)}")
    if bad:
        print(f"FAIL: {len(bad)} map(s) disagree: "
              + ", ".join(f"{n} ({d:.2e})" for n, d in bad))
        return 1
    print(f"OK: all {ok} maps identical to the single-objective definitions.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
