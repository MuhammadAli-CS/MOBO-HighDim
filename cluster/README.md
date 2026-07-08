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

## 4. Submit the experiments

This is the set that got killed running sequentially on a laptop
(correlation ablation's Kronecker-GP step and composite Penicillin were the
two RAM/disk-heavy ones). On the cluster these run **in parallel** as
independent jobs instead of one-at-a-time:

```
bash cluster/submit_all.sh
```

Check status with:
```
squeue -u $USER
```

Each job writes its own log to `cluster/logs/<label>_<jobid>.out`. Once a
group's jobs finish, a dependent plot job runs automatically
(`--dependency=afterok`) and regenerates `experiments/<exp>/comparison_seed0.png`
via `plot_comparison.py`'s auto-discovery — no manual replotting needed.

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
`run_experiment.sub` / `submit_all.sh` / `submit_llm.sh` if quota becomes an
issue.

## Pulling results back down

```
# from your laptop
scp -r <netid>@unicorn-login-01.coecis.cornell.edu:~/MOBO-HighDim/experiments/correlation_ablation_dtlz2curve ./experiments/
scp -r <netid>@unicorn-login-01.coecis.cornell.edu:~/MOBO-HighDim/experiments/penicillin_composite ./experiments/
```

Or just `git commit && git push` from the cluster checkout directly (mind
`.gitignore` — `experiments/**/*.pt` may be excluded; check before assuming
a push carries the result files).

## Adjusting resources

`cluster/run_experiment.sub` requests `--mem=16g --gres=gpu:1
--cpus-per-task=4 --time=08:00:00` for every job. The Kronecker-GP and
composite-Penicillin jobs are the two that blew up laptop RAM/disk
previously — if either gets OOM-killed or preempted-and-requeued
repeatedly on the cluster too, bump `--mem` first (try 32g), and consider
dropping `--gres=gpu:1` for the Kronecker job specifically if the cluster's
GPU nodes are memory-constrained rather than compute-constrained — check
`sacct -j <jobid> --format=MaxRSS,Elapsed,State` after a run to see which
limit was actually hit.
