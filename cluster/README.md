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
salloc --mem=8g --cpus-per-task=4 --time=01:00:00 --partition=kilian-interactive
bash cluster/setup_env.sh
exit   # end the interactive allocation once it finishes
```

This creates a conda env at `~/morbo-env` pinned to the same
torch/gpytorch/botorch versions validated locally (torch 2.12, botorch
0.9.5, gpytorch 1.11, Python 3.11 — see the main `README.md` "Fork notes"
section) and installs this repo into it with `pip install -e .`.

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

## 5. LLM-dependent parts (Parts 2 and 3)

```
export ANTHROPIC_API_KEY=sk-ant-...
bash cluster/submit_llm.sh
```

Confirm compute nodes (not just the login node) have outbound HTTPS access
before relying on this — some clusters firewall compute nodes off from the
public internet. If they don't, these two parts need to run from an
interactive login-node-adjacent session instead of a batch job.

## Partitions

Per Cornell's onboarding email: priority partition is `kilian` (private,
preemptive — reclaimed by the owner within ~1hr if contended), falling back
to the general `default_partition`/`gpu` community queues. Interactive jobs
submitted to `kilian` are automatically moved to `kilian-interactive`. All
scripts here default to `--partition=kilian`; change it in
`run_experiment.sub` / `submit_composite_curve_100d.sh` / `submit_llm.sh` if
quota becomes an issue.

## Pulling results back down

```
# from your laptop
scp -r <netid>@unicorn-login-01.coecis.cornell.edu:~/MOBO-HighDim/experiments/composite_curve_dtlz2_100d ./experiments/
```

Or just `git commit && git push` from the cluster checkout directly (mind
`.gitignore` — `experiments/**/*.pt` may be excluded; check before assuming
a push carries the result files).

## Adjusting resources

`cluster/run_experiment.sub` requests `--mem=16g --gres=gpu:1
--cpus-per-task=4 --time=08:00:00` for every job. The Kronecker-GP job here
is the one to watch — at d=20/200 evals it was already ~100x the
independent-GP fit cost (see `experiments/correlation_ablation_dtlz2curve/RESULTS.md`),
and this run is at 5x the input dimension and 3x the eval budget, so it's
the most likely one to need more memory or time. If it gets OOM-killed or
preempted-and-requeued repeatedly, bump `--mem` first (try 32g), and check
`sacct -j <jobid> --format=MaxRSS,Elapsed,State` after a run to see which
limit was actually hit.
