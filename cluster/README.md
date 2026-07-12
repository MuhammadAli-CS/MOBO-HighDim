# Running on Cornell's Unicorn cluster

One-time setup, then submit jobs from the login node — never run heavy
compute directly on the login node, it kills resource-intensive processes
automatically.

## 1. Connect

```
ssh <netid>@unicorn-login-01.coecis.cornell.edu
```

Requires being on-campus or on the Cornell VPN.

## 2. Get the code onto the cluster

```
git clone https://github.com/MuhammadAli-CS/MOBO-HighDim.git
cd MOBO-HighDim
```

## 3. Build the environment (one time only)

Do this from an interactive allocation, not the login node:

```
salloc --mem=8g --cpus-per-task=4 --time=01:00:00 --partition=default_partition-interactive --account=kilian
bash cluster/setup_env.sh
exit   # end the interactive allocation once it finishes
```

This creates a conda env at `~/morbo-env` (Python 3.11). `torch` is left
unpinned (the cluster's B200 GPUs need a CUDA build recent enough to
support Blackwell/compute capability 10.0, which the `cu121` channel used
locally doesn't have), but `botorch==0.9.5`/`gpytorch==1.11` stay pinned to
the exact versions this codebase's compatibility fixes were validated
against (see the main `README.md` "Fork notes" section — the port
specifically patches botorch APIs that changed between the archived
upstream's target, `~0.6`, and `0.9.5`). Installing a newer botorch
unpinned risks it having moved its API again since 0.9.5, the same class of
break the port already had to fix once. Installs this repo into the env
with `pip install -e .`.

## 4. Fig2-scale correlation-ablation follow-up (composite_curve_dtlz2_100d)

`correlation_ablation_dtlz2curve` (d=20) found only a ~0.3% composite-vs-direct
HV margin, much smaller than `fig2_dtlz2_100d`'s g/cos/sin composite's +25%
margin at d=100. `composite_curve_dtlz2_100d` reruns the same genuinely-
correlated 8-point curve construction at fig2's scale (d=100, 600 evals,
batch 50) to test whether that gap was a dimensionality effect. All three
labels (`morbo`, `independent_gp_composite`, `kronecker_gp_composite`) run
here — the Kronecker one specifically was moved off the laptop for this
config, since it was already ~100x the independent-GP fit cost at the
smaller d=20 scale and is expected to be substantially worse at d=100:

```
bash cluster/submit_composite_curve_100d.sh
```

Check status with:
```
squeue -u $USER
```

Each job writes its own log to `cluster/logs/<label>-100d_<jobid>.out`. Once
all three finish, a dependent plot job runs automatically
(`--dependency=afterok`) and generates
`experiments/composite_curve_dtlz2_100d/comparison_seed0.png` via
`plot_comparison.py`'s auto-discovery — no manual replotting needed.

## 4b. Trust-region shape adaptation (tr_shape_dtlz2_100d)

MORBO's trust regions were purely isotropic hypercubes — no per-dimension
lengthscale rescaling at all (a step below even the original TuRBO paper's
own technique). `tr_shape_dtlz2_100d` compares three alternative geometries
against the isotropic baseline at fig2 scale (d=100, 600 evals, batch 50):
`ard_box` (axis-aligned, rescaled by the TR's fitted GP ARD lengthscales),
`pca_ellipsoid` (rotated into the PCA frame of the TR's local data), and
`ard_pca_ellipsoid` (PCA rotation + lengthscale-reweighted axis widths).
See `morbo/trust_region.py`'s `TurboHParams.tr_shape` docstring for the
full design rationale (why a rotated *box* rather than a true ellipsoid,
why the isotropic baseline is provably unaffected, etc.):

```
bash cluster/submit_tr_shape_100d.sh
```

Same job-log/auto-plot pattern as `composite_curve_dtlz2_100d` above. The
isotropic baseline (`morbo`) is reused from a file already committed to the
repo (`experiments/tr_shape_dtlz2_100d/morbo/0000_morbo.pt`, identical to
`fig2_dtlz2_100d`'s own `morbo` result) rather than resubmitted — `git pull`
already has it, no job needed.

## 4c. New shape methods, robustness, and 2×2 (post-sweep follow-ups)

After the initial sweep, four more submission scripts were added (see
`writeup/FURTHER_DIRECTIONS.md` for the motivation and paper sources).
**Run `submit_smoke.sh` first** — it validates every new code path with a
tiny BO loop on the cluster before the multi-hour sweeps:

```
bash cluster/submit_smoke.sh          # ~5 min; wait for "All smoke tests passed."
bash cluster/submit_tr_shape_new_methods.sh   # cma_ellipsoid, linear-kernel, dim-prior fix @ d=100/150/200
bash cluster/submit_penicillin_2x2.sh          # composite modeling × shape adaptation, Penicillin
bash cluster/submit_tr_shape_multiseed.sh      # seeds 1–4 for the core methods (64 jobs — check sinfo first)
```

New methods (all reuse the existing `run_experiment.sub`, so the 16-CPU /
64 GB / 1-GPU / aimi / kilian config applies):
- `cma_ellipsoid` — CMA-ES covariance adaptation (AS-SMEA).
- `linear_gp` / `linear_gp_pca` / `linear_gp_cma` — spherically-projected
  linear kernel (linear-bo challenge baseline), alone and crossed with shape.
- `ard_box_dimprior` / `ard_pca_dimprior` — dimension-scaled lengthscale
  prior (Hvarfner) as a candidate fix for `ard_box`'s high-d collapse.
- `composite_penicillin_pca` / `_ard_pca` — composite modeling × shape.

After the multi-seed jobs land, aggregate with:
```
python aggregate_seeds.py tr_shape_dtlz2_100d   # mean ± std per method
```

## 5. LLM-dependent parts (Parts 2 and 3)

```
export ANTHROPIC_API_KEY=sk-ant-...
bash cluster/submit_llm.sh
```

Confirm compute nodes (not just the login node) have outbound HTTPS access
before relying on this — some clusters firewall compute nodes off from the
public internet. If they don't, these two parts need to run from an
interactive login-node-adjacent session instead of a batch job.

## Partitions and accounts

`kilian` is a SLURM **account**, not a partition — `scontrol show partition
kilian` returns "not found". The real partitions on Unicorn are
`default_partition`, `gpu`, `spark`, and `aimi` (each with a `-interactive`
variant with a 2-day time limit, for `salloc`). Your priority (per Cornell's
onboarding email) comes from the `kilian` **account**, not a partition name —
confirmed via `sacctmgr -p show assoc user=<netid>`, which lists `kilian` as
your account.

All scripts here therefore use `--partition=aimi --account=kilian` for actual
BO runs, and `--partition=default_partition --account=kilian` for the
lightweight plot jobs that don't need a GPU. `aimi` is the SURP program's
own partition (`aimi-compute-[01-03]`: 224 CPUs / ~2TB RAM / 8x NVIDIA B200
per node) — access is gated by the `en-cc-unicorn-aimi-users` group
(`scontrol show partition aimi`'s `AllowGroups`), confirmed present via `id`,
with `AllowAccounts=ALL` so the existing `kilian` account works there too.
This is meaningfully more powerful than the general `gpu` partition (whose
best nodes top out at RTX A6000), so it's used in preference to `gpu` for
every job here. Check current node load with `sinfo` before submitting if
you want to avoid queueing behind fully-`alloc`'d nodes — GPU
type isn't pinned by default, so jobs land on whatever's free rather than
waiting for a specific card.

## Pulling results back down

```
# from your laptop
scp -r <netid>@unicorn-login-01.coecis.cornell.edu:~/MOBO-HighDim/experiments/composite_curve_dtlz2_100d ./experiments/
```

Or just `git commit && git push` from the cluster checkout directly (mind
`.gitignore` — `experiments/**/*.pt` may be excluded; check before assuming
a push carries the result files).

## Adjusting resources

`cluster/run_experiment.sub` requests `--mem=64g --gres=gpu:1
--cpus-per-task=16 --time=08:00:00` for every job. The Kronecker-GP job here
is the one to watch — at d=20/200 evals it was already ~100x the
independent-GP fit cost (see `experiments/correlation_ablation_dtlz2curve/RESULTS.md`),
and this run is at 5x the input dimension and 3x the eval budget, so it's
the most likely one to need more memory or time. If it gets OOM-killed or
preempted-and-requeued repeatedly, bump `--mem` first (try 32g), and check
`sacct -j <jobid> --format=MaxRSS,Elapsed,State` after a run to see which
limit was actually hit.
