# MOBO-HighDim: Related Work & Direction Notes

Carried over from the SURP color-mixing benchmark project's literature review session.
This is a working reference for the group's next steps: high-dimensional / composite
multi-objective Bayesian optimization, and where LLMs might fit in.

## Team next steps (as of this write-up)

- **Ricky**: LLM-assisted trust-region BO for one composite function.
- **Sahas**: Explore a categorized (space-partitioned) multi-objective architecture.
- **Muhammad**: Literature review + identify matching baselines/benchmarks.

Two concrete directions identified so far:
- **MORBO with composite functions** — nobody has combined composite-objective
  modeling (GP on a raw response + known reduction) with MORBO's scalable
  coordinated-trust-region machinery. See "Open gap" section below.
- **LLM-assisted MORBO** — bridges Ricky's and Sahas's tasks into one deliverable;
  no paper reviewed so far puts an LLM inside MORBO's specific mechanism
  (coordinated TR-center selection + shared local models + hypervolume batch selection).

---

## Paper write-ups

### Trust-Aware LLM-Assisted BO
Zhou, Wang, Gu & Tan · *Integration, the VLSI Journal* (2026)

Chooses "specifications" (targets to move toward) via LLM or engineer. Runs a
separate BO loop per specification; within each, models different objectives
independently with different GPs, then combines the best designs from each
specification's run into a Pareto front at the end.

Algorithm per specification:
- LLM proposes initial designs.
- Reparametrize each objective as `g_i(x) = sqrt(2(f_i(x) - f_i*))`, where `f_i*`
  is that objective's target value — `g_i(x) = 0` means "hit the target exactly."
  Train one GP per `g_i`.
- **Expected Regret (ER)** combines the `M` independent GPs into one acquisition
  decision, picking the next `x` that minimizes aggregated regret jointly across
  all objectives — not the best for any single GP, but the best compromise across all.
- **Trust probability** `p_t := λ/t²` — LLM is trusted more early on, decaying over
  time. A Bernoulli draw picks LLM vs. ER's own candidate each round; if the LLM
  candidate is chosen, it's still screened through the ER acquisition function to
  make sure it isn't bad, otherwise ER's own candidate is used instead.
- LLM also assists kernel construction (group-importance weighting, parameter grouping).

*Caveat: this write-up is derived from earlier review notes, not an independent
read of the full paper — worth verifying directly if precision matters.*

---

### LLMs for BO in Scientific Domains: Are We There Yet?

Finds that LLM-only optimizers mostly lean on pretrained knowledge rather than
actually using the experimental history provided — tested by feeding a BO loop
harmful fake data in one run and true data in another, and observing similar
performance either way.

Proposes **LLMNN** (no Bayesian optimization at all):
- LLM chooses promising cluster centers to search from.
- Nearest-neighbor search around each center fills out a batch of unexplored candidates.
- Batch is evaluated, results appended to history, repeat.

Good performance vs. current methods despite the lack of any BO machinery — a
useful low-effort baseline for isolating "is the BO apparatus actually pulling
its weight, or is cluster+NN good enough?"

---

### LAGO — Combining Trust Region Methods and Bayesian Optimization
Van Dieren, Vanzan & Nobile · arXiv:2603.02970 (2026), EPFL / Politecnico di Torino

Combines global BO (gradBO) with local gradient-based trust-region refinement
(SR1TR). Every iteration, the two proposals **compete** rather than switching on
a fixed schedule or failure trigger.

Algorithm per iteration `t`:
- **Global step (gradBO)**: fit a GP over both function values and gradients at
  all evaluated points; optimize EI outside the current trust region → candidate `x_G`.
- **Local step (SR1TR)**: build a quadratic model of `f` around the trust-region
  center, minimize it within the region → candidate `x_L`. Curvature (Hessian)
  updated via SR1, not BFGS, since BFGS forces positive-definiteness — the wrong
  assumption if the local region is non-convex.
- **Selection rule**: accept the global candidate if `EI(x_G) > γ·I_t` (`I_t` = local
  step's predicted improvement); otherwise evaluate `x_L`. One evaluation per iteration either way.
- **Trust-region resizing**: compare actual vs. predicted improvement after evaluating
  `x_L` — accurate prediction expands the region, inaccurate shrinks it.
- **Data admission filter**: only fold an evaluated local point into the global GP
  if it's farther than `ν·ℓ` from every existing point (keeps the shared GP well-conditioned).

**Key limitation (stated directly by the authors, and important for our use case):**
requires **noise-free, differentiable** objective evaluations — the trust-region
component needs exact function *and gradient* values. Best suited to low-to-moderate
dimensional problems where gradients are cheap relative to dimension (e.g.
adjoint-based PDE-constrained optimization); gradient-enhanced GPs scale cubically
and get expensive fast as dimension grows. **This rules LAGO's SR1TR component out
for most of our candidate benchmarks** (Penicillin's ODE integrator, mixbox, etc.
are not differentiable) — any LLM-assisted TR work should stay gradient-free,
closer to TuRBO/MORBO than to LAGO.

Passed: multimodal (Styblinski-Tang) and PDE-constrained problems, where
local-only methods and LABCAT got stuck in the wrong basin. Weaker: ill-conditioned
(Rosenbrock — LABCAT's PCA-aligned region wins) and highly multimodal (Levy — spends
most of the budget in global mode).

---

### Composite Bayesian Optimisation for Multi-Objective Problems with Smooth Tchebycheff Scalarisation
Nogueira Pires, Cardoso Coelho & Andrade Pires · *Structural and Multidisciplinary
Optimization* (2026) — [Springer link](https://link.springer.com/article/10.1007/s00158-025-04229-y)
(paywalled; SSRN preprint blocked by bot-check — not independently re-verified beyond the underlying STCH method below)

Composite structure: `J(θ) = g(L(θ, Y(θ)))`
- `Y(θ)` — the raw response curve. This is the actual black box the GP predicts
  (via a PCA-compressed representation).
- `L(·)` — a known, analytic reduction operator (e.g. "area under the
  moment-rotation curve" → energy). Takes the curve, produces the raw objectives `f_i`.
- `g` — **smooth Tchebycheff (STCH) scalarisation**. Takes the objectives, produces
  the final scalar being optimized *for one specific preference vector `λ`*.

**How it finds a Pareto front** (confirmed directly from the underlying STCH paper,
Lin & Zhang, arXiv:2402.19078, which this method builds on): STCH is a
*per-preference-vector* solver, not a whole-front algorithm. `"once a specific
preference λ is given, we can directly optimize the STCH scalarization with a
straightforward gradient descent algorithm."` Front coverage comes from solving
this repeatedly with different sampled `λ` vectors and pooling the resulting optima
— the same strategy MORBO's related work attributes to ParEGO/TS-TCH. Theorem 3.8
in the base STCH paper guarantees every Pareto-optimal point (including
non-convex/concave regions, unlike plain weighted-sum scalarization) is reachable
by *some* valid `λ` — but the base paper does **not** prescribe a `λ`-sampling
strategy; that's left to the practitioner. Smoothing (`μ` parameter) fixes the
classical Tchebycheff max's non-differentiability so gradient-based acquisition
optimization actually converges.

Three-way comparison in the paper: classical BO vs. composite BO vs. "full composite
BO" (nested composite BO).

Two examples:
- **16-layer foam beam design**: full composite and regular composite both converge
  in ~75 evals vs. 200+ for classical BO — no real edge from going full composite
  here, problem not complex enough at the composition level.
- **Polycrystalline hardening parameter fitting**: full composite clearly wins
  (~20 evals vs. unconverged classical/composite within budget) — this is where the
  nested structure actually pays off.
- Trade-off: full composite costs way more compute time (835 min vs. 126 vs. 44 min
  on the beam problem) — worth it only when the objective is genuinely
  nested/complex, not free efficiency.

**Not high-dimensional** — like BoTier below, this operates in the standard
low-dimensional expensive-experiment regime, no trust regions or space
partitioning. Reinforces that composite-BO methods generally haven't been
combined with high-dimensional scaling machinery.

---

### MORBO — Multi-Objective Bayesian Optimization over High-Dimensional Search Spaces
Daulton, Eriksson, Balandat & Bakshy · UAI 2022 ([arXiv:2109.10964](https://arxiv.org/pdf/2109.10964)) ·
Official code: [github.com/facebookresearch/morbo](https://github.com/facebookresearch/morbo) (archived Oct 2023, read-only)

Optimizes diverse parts of the global Pareto frontier in parallel using a
coordinated set of local trust regions (TRs), fixing the poor coverage from
naively extending TuRBO to multi-objective (independent TRs each optimizing
their own random Chebyshev scalarization — see "Issues with scalarized TuRBO" below).

**Genuinely multi-objective** — not a scalarization trick. Explicitly optimizes a
vector-valued `f(x) = [f⁽¹⁾,...,f⁽ᴹ⁾]`, tracks Pareto dominance, and its acquisition
target (hypervolume improvement) is defined over the actual Pareto frontier.

**Does not need gradients.** Fully black-box/derivative-free w.r.t. the true
objective — local GPs are standard regression on `(x, f(x))` pairs. The only
"gradients" involved are the standard BoTorch practice of optimizing the
*acquisition function* (built on the differentiable GP posterior) via L-BFGS —
that's true of virtually all modern BO, not a special requirement. This is why
MORBO (unlike LAGO) is usable on non-differentiable simulators like Penicillin's
Euler-loop ODE integrator.

Algorithm at each iteration `t`:
- Fit a **local GP within each trust region**, using all observations (from *any*
  TR) that fall inside a padded hypercube (`2L` edge length) around that TR — not
  just points that TR itself collected. Keeps GP inference cheap even at large
  evaluation budgets. (Note: TuRBO *already* runs multiple independent TRs each
  with their own GP — that part isn't new. The new part is that MORBO's TRs share data.)
- Select a batch of `q` candidates via **sequential greedy hypervolume improvement
  (HVI)**: draw Thompson-sampled posterior realizations from each TR's local GP,
  pool all TRs' candidates together, and pick each batch slot to maximize hypervolume
  improvement jointly with slots already chosen — this is what makes the TRs
  *collaborate* on one shared objective instead of acting independently.
- Update each TR's success/failure counters based on the **shared global HV
  utility** (not the TR's own local objective) — a TR only counts a success if one
  of its candidates actually expanded the global Pareto frontier.
- Grow/shrink each TR's edge length the same way TuRBO does; terminate once length
  drops below a minimum.
- **Global search for new centers**: when a TR terminates, fit a *separate* GP
  only on the history of past TR center/restart points (much smaller dataset than
  the full observation history) — not the same as the per-TR local GPs. Sample a
  random objective-weighting `λ` and one posterior draw from this small GP, find
  `argmax` of the scalarized draw over the **entire global domain**, evaluate that
  point for real, reinitialize the dead TR there. Because `λ` is resampled randomly
  every restart, this is what actively spreads new TRs across different tradeoff
  directions instead of letting them cluster.
  - *(While a TR is still alive, its center just moves to whichever already-evaluated
    point in its own data has the highest hypervolume contribution — cheap, no
    global search needed. The expensive global-search step above only fires on
    termination.)*

**Issues with scalarized TuRBO** (their §2.3, the direct TuRBO comparison): the
"obvious" MOO extension of TuRBO — multiple TRs, each optimizing its own random
Chebyshev scalarization — gives very poor Pareto coverage, since a TR doing well
on "its" scalarization just keeps running and few scalarizations end up being
explored. Demonstrated qualitatively (their Figure 2) on synthetic **DTLZ2 (d=100)**
and **MW7** (d=10, 2 constraints) — TuRBO+scalarization clusters around a few
points, MORBO spreads across the whole front and finds disconnected regions on MW7.
**This is not part of the paper's main quantitative benchmark suite** — it's an
illustrative motivating comparison, not a rigorous head-to-head with error bars.

Main quantitative results (Figure 3): **Rover trajectory planning** (d=60, public,
Wang et al. 2018), **AR/VR optical display design** (d=146, hours per proprietary
sim — **not runnable by us**), **Mazda 3-vehicle design** (d=222, 54 constraints,
original solve took ~3,000 CPU-years — **not runnable by us**), plus smaller
vehicle/welded-beam/DTLZ3/5/7 problems in the appendix (framed as "MORBO is
competitive... on problems it was not designed for," not dominant there).
Baselines: qNEHVI, qParEGO, TS-TCH, TSEMO, DGEMO, MOEA/D-EGO, LaMOO-CMAES,
LaMOO-qNEHVI, NSGA-II, Sobol.

**Result**: on trajectory planning and optical design, even qNEHVI doesn't beat
plain NSGA-II — only MORBO clearly wins, because its local modeling is the only
approach that scales enough to use the evaluation budget effectively at these
dimensionalities. On Mazda, Sobol couldn't find a second feasible design after
the initial one; NSGA-II made progress but wasn't competitive with MORBO.

**Is it "strictly better than TuRBO"? No — scoped claim.** MORBO solves a
different problem (multi- vs. single-objective); for single-objective problems
there's no Pareto front to spread across, so all of MORBO's coordination
machinery has nothing to do. It's strictly better than the *naive multi-TR +
scalarization* extension of TuRBO specifically, and best in the
**high-dimensional, large-evaluation-budget regime** — their own appendix frames
small-scale results as "competitive," not dominant; standard qNEHVI/qParEGO can
match or beat it at low dimension without MORBO's added complexity (hypervolume
computation cost scales badly with number of objectives; extra restart-search
machinery TuRBO doesn't need).

**Open gap — MORBO of composite functions**: MORBO's local GPs model the
objectives directly, not a raw intermediate response + known reduction. Grafting
composite modeling on: each local GP would model the raw response `Y(x)` instead
of `f(x)` directly; HVI would need to be computed by pushing posterior *samples*
of `Y(x)` through the known reduction before computing hypervolume (Monte Carlo,
but cheap — deterministic function applied to samples you already have); the
TR-center HVC calculation and the restart-time global-search step would need the
same treatment. Maps almost directly onto our own `ResidualBOCFSolver` (which
already does composite-GP + known-objective-inside-acquisition for a single
region) — "give it MORBO's multi-region coordination" is a smaller lift than
building either piece from scratch. **Hypervolume and "composite" are orthogonal
design axes, not competing choices** — hypervolume answers "how do I score
candidates given objective values I already have," composite modeling answers
"should the GP learn the objectives directly or a raw response that a known
formula converts into them." Nothing stops combining both.

**Practical note for benchmarking against TuRBO ourselves**: official repo ships
DTLZ2/3/5/7, Rover, VehicleSafety, WeldedBeam pre-wired (`python main.py
<experiment_name> <algorithm> <seed>`) but has **no TuRBO baseline pre-configured**
— would need to implement the naive scalarized-TuRBO strawman ourselves (matches
their own Figure 2 comparison). Repo is archived/unmaintained since Oct 2023 —
recommend an isolated venv, not our main project's `.venv`, since dependency
versions may need pinning. For a fast pitch demo (not full 60D/222D scale):
VehicleSafety (d=5) or a small DTLZ2 would show the same qualitative story
(poor coverage from scalarized-TuRBO vs. good coverage from MORBO) fast enough
to run live.

---

### MOHOLLM — Multi-Objective Hierarchical Optimization with LLMs
Schwanke, Ivanov, Salinas, Hutter & Zela · arXiv:2601.13892 (2026), University of
Freiburg / ELLIS Institute Tübingen

Uses LLMs as both candidate sampler and surrogate model inside a structured
hierarchical search, adaptively partitioning the input space into disjoint
hyperrectangular regions and ranking them with a composite score, so the LLM only
has to reason locally within a promising sub-region rather than about the global
structure of the problem.

Algorithm at each iteration `t`:
- **Adaptive space partitioning**: build a KD-tree over the observation history,
  splitting along the dimension of maximal variance at the median value; a leaf
  only gets refined once its sample count exceeds a time-growing threshold
  (broad exploration early, finer refinement later).
- **Region scoring**: composite utility per leaf = (i) Pareto exploitation —
  regional hypervolume contribution to the current Pareto front, (ii) geometric
  exploration — region volume (keeps large under-sampled regions competitive),
  (iii) statistical exploration — variance-based UCB on each region's HV contribution.
- **Stochastic region selection**: softmax over composite scores (not greedy) —
  guarantees every region gets selected infinitely often, which their
  Pareto-consistency proof needs.
- **LLM-based candidate sampling**: prompt the LLM to generate `N` candidates
  within each selected region, conditioned on that region's local history.
- **LLM-based surrogate evaluation**: LLM predicts objective values for all
  generated candidates, batch that maximizes predicted hypervolume gets selected
  *before* spending a real evaluation.

Proves Pareto consistency almost surely under standard regularity assumptions
(Lipschitz objectives, shrinking partitions, non-starvation region selection,
non-degenerate LLM sampling) via a Hausdorff-distance convergence argument.

Tested on 12 synthetic (DTLZ1-3, BraninCurrin, ChankongHaimes, GMM, Poloni,
SchafferN1-2, TestFunction4, ToyRobust, Kursawe) + 3 real-world: **Penicillin**
(d=7, M=3), **VehicleSafety** (d=5, M=3), **CarSideImpact** (d=7, M=4). Baselines:
qLogEHVI/EHVI, NSGA-II, NSGA-III, MOEA/D, GDE3, plus a **"global LLM" ablation**
(same prompts/evaluation, no space partitioning) — the key control for isolating
what the partitioning itself buys you.

**Result**: beats the global-LLM baseline, competitive with MOEAs, but does
**not** surpass standard MOBO (qEHVI family) on sample efficiency at small
budgets — stated directly by the authors. Notable ablation: LLM-as-surrogate got
Spearman 0.874 / R² 0.757 on real-world tasks; GP and TabPFN-v2 surrogates showed
near-zero or negative correlation there.

Limitations (stated directly): axis-aligned KD-tree partitioning doesn't extend
cleanly to non-Euclidean domains; LLM inference cost grows with dimensionality,
limiting use at large budgets; formal hypervolume regret bounds remain open
(only Pareto-consistency is proven).

**Core mechanism is essentially the same as LEMOE** (see below) — LLM fully
replaces the GP as both sampler and surrogate, hypervolume-driven acquisition
computed from LLM point-predictions rather than a true GP posterior. MOHOLLM
adds hierarchical space partitioning + a convergence proof and is domain-general;
LEMOE has neither but adds a domain-specific (LLVM-based) warm-start.

---

### LEMOE — LLM-Enhanced Multi-Objective Bayesian Optimization for Microarchitecture Exploration
Li, Zhang, Li, Yin & Wang · DAC 2025, Fudan University

LLM-enhanced multi-objective BO for RISC-V (BOOM core) microarchitecture design
space exploration — tuning ~25 architecture parameters (cache sizes, decode
width, issue widths, queue depths, etc.) to jointly maximize IPC and minimize
power, evaluated via expensive hardware simulation (hours per design).

Two LLM insertion points:
1. **Program-aware warm-start**: LLVM analyzes the target workload's IR to
   extract program features (basic block count, memory ops, branch density,
   etc.); LLM uses those features to propose an informed initial design set.
2. **LLM fully replaces the GP** — no Gaussian process anywhere. LLM is prompted
   with the observation history as text and asked to (a) propose new candidate
   configurations and (b) predict their IPC/power directly. EHVI is computed
   non-analytically from these LLM-predicted values, sidestepping EHVI's usual
   expensive computation and the need for explicit GP uncertainty estimates.

Results: 22.8–45.3% better hypervolume/IPC-power tradeoff, 2–3.6× faster
runtime-to-target vs. prior RISC-V DSE methods (Random Forest, TPE,
BOOM-Explorer, ArchGym, an existing LLM-BO baseline). Ablations show
RISC-V-specific domain knowledge in the LLM's prior is the single biggest
contributor — bigger than either the LLVM warm-start or the LLM sampler alone.

**Relevant for Sahas**: closest thing found yet to fully dropping the GP in
favor of an LLM surrogate (not just augmenting one) — a data point on how far
"no GP at all" can go before losing principled uncertainty quantification.

---

### AutoLead — LLM-Guided Bayesian Optimization for Multi-Objective Lead Optimization
Zhang, Choong & Ozawa · bioRxiv 2025.08.19.671029 (2025), University of Tokyo / SB Intuitions

Combines an LLM's chemical reasoning with GP-based BO for drug lead
optimization, using a transient hybrid schedule (LLM-heavy early, BO-heavy late)
rather than a fixed split. Solves "BO can't optimize over discrete SMILES
strings" by BO-ing in a continuous normalized molecular-descriptor space and
using the LLM itself as the inverse decoder back to a valid molecule.

**The LLM-as-decoder trick, expanded**: BO needs a continuous space to optimize
an acquisition function over, but a molecule (SMILES string) is discrete. Fix:
- Forward map `φ`: molecule → 10-dim vector of normalized RDKit descriptors (MW,
  logP, TPSA, HBD/HBA counts, rotatable bonds, ring counts, etc.) — easy, deterministic.
- BO operates entirely in this continuous descriptor space (GP + UCB), producing
  an "ideal" target descriptor vector.
- The hard part: no closed-form inverse `φ⁻¹` exists (can't go from "I want these
  properties" to an actual valid structure).
- **The LLM stands in for the missing inverse.** Verbalize the target vector as a
  sentence ("molecular weight: 310 Da, logP: 3.5, ...") and prompt the LLM to
  generate a matching SMILES string (`φ̃⁻¹` — tilde marks it as an approximate
  stand-in). No dedicated generative model (VAE/graph-decoder) needed, no
  domain-specific fine-tuning.

Algorithm at each iteration `t`:
- Warmstart: LLM proposes a diverse initial pool of valid molecules retaining the
  reference compound's core scaffold.
- Fit GP on `U(f(s))` (weighted-average utility) over the 10-dim descriptor space.
- `z_t ~ Bernoulli(p_t)`, `p_t = min(t^α/T, 1)` — probability of using BO grows
  over the run (LLM-heavy early → BO-heavy late).
- If BO selected (and ≥3 points collected): optimize UCB in descriptor space,
  decode via LLM. If LLM selected: LLM proposes directly from chemical priors.
- Evaluate true properties, update dataset and running Pareto front.

Tested on **ChatDrug-200** (28 single+multi-objective property-editing tasks),
**DrugAssist-500** (interactive multi-property editing), and their new
**LipinskiFix-1000** (1,000 real ligands from HiQBind violating Lipinski's Rule
of Five — restore compliance while raising QED, a more realistic scenario than
the synthetic tasks). Baselines: Random, PCA, High-Variance, GS-Mutate,
MoleculeSTM-SMILES, ChatDrug (GPT-3.5-Turbo / GPT-4o).

Result: large consistent margins, most dramatic on strict multi-objective
thresholds — e.g. LipinskiFix-1000: AutoLead-HO 28.9% vs. MoleculeSTM 13.9% vs.
ChatDrug 3.87%.

**Shares its core schedule idea with Trust-Aware LLM-Assisted BO** (both use a
monotonic Bernoulli-sampled probability shifting from LLM-favored to
optimizer-favored over time — AutoLead cites this from prior work, doesn't claim
it as novel). **Diverges**: (1) AutoLead fits **one GP on a scalarized utility**,
not a separate GP per objective + Expected Regret — not really a multi-objective
architecture the way Trust-Aware BO is; (2) no acquisition-based screening of LLM
candidates before acceptance, unlike Trust-Aware BO's validation step. AutoLead's
actual distinct contribution is the SMILES/descriptor-space decoder trick, not the schedule.

Limitation stated directly: the feature-to-SMILES inverse mapping (LLM-as-decoder)
is the identified weak link — its fidelity bounds how well an optimized
descriptor point translates back to the intended molecule.

---

### BoTier — Multi-Objective Bayesian Optimization with Tiered Composite Objectives
Haddadnia, Grashoff & Strieth-Kalthoff · arXiv:2501.15554 (2025), Harvard / University of Wuppertal

**Genuinely composite (Astudillo–Frazier sense), but deliberately *not*
hypervolume-based** — worth being precise about, since composite-modeling and
hypervolume/EHVI are two orthogonal design axes, and BoTier explicitly picks
scalarization over hypervolume rather than combining them.

- Each raw objective `ψᵢ` (yield, cost, temperature — the actual measured
  quantities) gets its own independent GP.
- **Hierarchical scalarization** `Ξ = Σ(min(ψᵢ, tᵢ) · ∏H(ψⱼ − tⱼ))` combines them
  into **one single scalar score**, computed via Monte Carlo over the GP
  posteriors (not point predictions) — the textbook composite pattern.
  Satisfaction thresholds `tᵢ` create a tiered preference: objective `i` only
  contributes once all higher-priority objectives clear their thresholds.
  Differentiable approximations of `min` and the Heaviside step function `H`
  make it auto-differentiable (PyTorch/BoTorch-native).
- Once you have the one scalar `Ξ(x)`, run **ordinary single-objective BO** on
  it (e.g. qEI) — no Pareto front is mapped, no hypervolume computed at all.
- Explicit motivation: EHVI/hypervolume methods spend budget mapping parts of
  the Pareto front the researcher doesn't actually care about. If priorities are
  already known (tiered thresholds), that coverage is wasted effort. **EHVI is
  one of BoTier's baselines**, not a component of BoTier's own method.
- Extends prior work called Chimera (Aspuru-Guzik group) by making the
  scalarization auto-differentiable and batch-evaluable (Chimera's original
  formulation required all K observations at once, blocking batch evaluation).

Benchmarks: 4 analytical BoTorch surfaces (BNH, DH4, DTLZ5, ZDT1) augmented with
secondary input-dependent objectives, plus real chemistry (Suzuki coupling,
benzylation, enzymatic alkoxylation, nanoparticle synthesis). Baselines: Chimera,
EHVI, penalty-based scalarization, Sobol.

Result: BoTier (even used as a black-box objective) converges faster to the
optimal `Ξ` than Chimera; using it as a *composite* objective further accelerates
optimization — including, surprisingly, in problems where all objectives are
purely output-dependent (not just the input-dependent ones the composite
structure was designed for), suggesting it's just easier to model several
independent distributions than one complex joint one.

**Not high-dimensional.** All benchmarks are low-dim (2–6 typical reaction
parameters) — global GP per objective, no trust regions, no partitioning. Same
category as the STCH composite paper above: confirms rather than fills the
"composite + high-dimensional" gap.

---

### MOBO_aircraft — Multi-Objective Bayesian Optimization of Composite Aircraft Wings Using Various Carbon Fibers
Liu, Abe, Kano, Yatsu, Nakamura, Shimoyama, Okabe & Obayashi · *Composite
Structures* 383 (2026) 120105

**Naming collision worth flagging** — "composite" here means composite
*materials* (carbon-fiber-reinforced plastic), **not** composite *functions*.
This is a standard (non-composite-BO) multi-objective BO application.

Real aerospace engineering problem: two-way coupled aeroelastic (CFD) +
structural sizing (FEM) simulation of a CFRP aircraft wing. Two Kriging (GP)
surrogates model **drag coefficient** and **wing weight** directly (not a raw
intermediate response — these are the actual objectives). Acquisition:
**EPBII** (Expected Penalty-Based-boundary-Intersection Improvement) —
PBI-style scalarization, similar family to ParEGO. 5 design variables
(chord/span/sweep geometry parameters). Compares MBO vs. NSGA-II, and separately
studies the effect of 3 carbon fiber types (T700S, T800S, T1100G) on the
resulting Pareto-optimal wing designs, plus the effect of manufacturing fiber
misalignment on compressive strength/weight.

Result: MBO produced a more diverse, more advanced Pareto front using ~1/10th
the function evaluations of NSGA-II.

**Good candidate as a new real-world benchmark problem** (aerostructural
design — a domain not covered by anything else on our list), small enough
(5D, 2-objective) to be tractable, though not itself a composite-BO or
high-dimensional method paper.

---

## Follow-up review: trust-region shape adaptation (2026-07-12)

Targeted search after building `tr_shape` (isotropic/ard_box/pca_ellipsoid/
ard_pca_ellipsoid/cma_ellipsoid/mab_shape, see
[`writeup/PROJECT_HANDOFF.md`](writeup/PROJECT_HANDOFF.md)) — looking for
prior art on shape-adaptive trust regions specifically, better bandit
designs for `mab_shape`, and real benchmarks to validate the
effective-dimension finding beyond our own synthetic `SparseDTLZ2`.

### Directly overlapping prior art (must cite / differentiate from)

**LABCAT** — Visser, van Daalen & Schoeman, arXiv:2311.11328 (2023, rev. 2024).
Rotates the trust region to align with weighted principal components of
local data *and* rescales axes by GP ARD lengthscales — essentially the
union of our `pca_ellipsoid` and `ard_box` ideas, single-objective only,
tested on COCO/BBOB. No effective-vs-nominal-dimension analysis, no MO
extension, no bandit-over-shapes. **This is the closest single prior-art
match to our work and needs an explicit differentiation paragraph in
`methods.tex`**: our contribution is (a) the MORBO/multi-objective setting,
(b) the effective-dimension-vs-budget governing-variable result backed by
`SparseDTLZ2`, (c) `cma_ellipsoid` as a persistent-covariance alternative
with a distinct low-data regime where it wins, and (d) `mab_shape` as a
portfolio mechanism over shape families, none of which LABCAT has.

**CMA-BO / CMA-TuRBO / CMA-BAxUS** — Ngo, Ha, Chan, Nguyen & Zhang, TMLR 2024,
arXiv:2402.03104. Uses CMA-ES as a *meta-algorithm* estimating a
distribution over where the optimum likely lies, then restricts an existing
method (BO/TuRBO/BAxUS) to that region — a softer "focus region," not an
ellipsoidal trust region built directly from CMA's rank-mu/evolution-path
covariance the way our `cma_ellipsoid` is. Confirms CMA+TuRBO is an active,
validated combination (supports the motivation) but is mechanistically
distinct enough to just need a differentiating footnote, not a rewrite.

**FuRBO** — Ascia et al., AutoML 2025, arXiv:2506.14619. Replaces TuRBO's
hyperrectangle with a hypersphere for constrained BO — still isotropic, not
anisotropic. Useful as "the field agrees hyperrectangle shape is a
weakness," but doesn't go as far as ellipsoid/PCA/CMA.

**AdaScale-TuRBO** — Tang & Paulson, arXiv:2604.22967 (2026),
github.com/PaulsonLab/AdaScale-TuRBO. Scales the GP lengthscale prior
*jointly* with both dimension and current trust-region size, rather than
just dimension (as the Hvarfner prior we already tried does). **Concrete
follow-up experiment, not just a citation**: worth checking whether this
joint scaling — unlike the static Hvarfner-style prior we tried — actually
rescues `ard_box`'s collapse. If it does, our negative result gets a
nuance; if it still fails, that's an even stronger case that the failure
is structural (treating `d` lengthscale estimates as `d` separate hard
constraints), not a lengthscale-*value* artifact at all.

**Regional Expected Improvement (REI)** — Namura & Takemori, arXiv:2412.11456
(2024). Not shape adaptation — a region-averaged acquisition function for
choosing *which* trust region to explore next (orthogonal to our "what
shape is each region" question; the two could compose). Their benchmark
set includes a Human-Powered Aircraft design family (d=17/32/32/108, real
engineering) — see benchmark table below.

MTRBO (arXiv:2605.06618) and LAGO's cousin-paper family confirm trust-region
BO is an active area generally, but neither addresses geometry — lower
priority, citation-list only.

**Bottom line**: no paper combines (a) a bandit over multiple trust-region
*shape families*, (b) in a multi-objective setting, (c) with an explicit
effective-dimension-vs-budget theory of when it helps. That combination
remains genuinely open territory for us.

### Better bandit designs for `mab_shape` (currently plain epsilon-greedy)

**GP-Hedge** — Hoffman, Brochu & de Freitas, UAI 2011, arXiv:1009.5419. The
foundational "bandit over BO strategies" paper — treats acquisition-function
selection as an adversarial bandit, uses Hedge/EXP3-style exponential
weights with a regret bound, not epsilon-greedy or UCB. Good origin
citation for `mab_shape`'s framing, and Hedge/EXP3 (softmax over cumulative
reward) is a concrete ablation against our epsilon-greedy choice.

**Self-tuning portfolio-based BO** (ScienceDirect, 2021) — replaces
GP-Hedge's hand-tuned hyperparameters with **Thompson sampling** over the
portfolio (Beta posterior per arm, sample-and-pick). Directly actionable:
a Thompson-sampling `mab_shape` variant removes the `mab_epsilon`
hyperparameter entirely.

**PCR-BO** (2024 book chapter) — a recent bandit-portfolio method whose
reward signal weighs recent progress with a discount/window, not just the
instantaneous success/fail we use. Suggests enriching `mab_shape`'s binary
success-streak reward with a windowed or magnitude-weighted signal
(e.g. hypervolume-improvement size, not just sign).

**Contextual bandits** (general HPO literature, e.g. LinUCB-style). The
most promising upgrade path specifically for us: since our own finding is
that *effective dimension relative to budget* governs which shape wins,
that's a natural context feature (a cheap online estimate — e.g. the ratio
of top-k PCA eigenvalue mass to total variance in a TR's local data, which
`pca_ellipsoid` already computes) for a contextual bandit replacing pure
trial-and-error. This would turn `mab_shape` from "epsilon-greedy recovers
the best of both worlds" into "the bandit learns exactly the rule we
discovered" — a substantially stronger result.

### Other SOTA HDBO methods as potential baselines/arms

| Method | What it does | Fit for us |
|---|---|---|
| SAASBO (Eriksson & Jankowiak, UAI 2021) | Sparse axis-aligned GP prior (half-Cauchy on lengthscales), automatically discovers active dims | Baseline, conceptually adjacent to our effective-dim story; not a TR-shape arm |
| BAxUS (Papenmeier et al., NeurIPS 2022) | Nested random embeddings, adaptively grows subspace dim inside a TuRBO-like loop | Could be a fixed `mab_shape` arm: "embed into growing random subspace" |
| Bounce (Papenmeier, Nardi, Poloczek, NeurIPS 2023, arXiv:2307.00618) | Extends BAxUS-style embeddings to mixed continuous/discrete spaces | Future mixed-variable extension, not core now |
| Vanilla-BO w/ dim-scaled prior (Hvarfner et al., ICML 2024) | Already tried as an `ard_box` fix (didn't work) | Missing as a **standalone global-BO baseline** (no trust region at all) |
| Linear-kernel + boundary-avoiding transform (Doumont et al., AISTATS 2026, arXiv:2512.00170) | Already tried (`linear_gp`); worth double-checking our transform matches their exact geometric fix | Verify implementation detail against the paper |
| Automated Kernel Discovery (2026, arXiv:2605.20249) | LLM-driven evolutionary search over GP kernel forms | Different axis (kernel search vs. TR geometry); citation only |

### Real benchmarks to validate the effective-dimension finding beyond `SparseDTLZ2`

Our strongest current result (§7, `experiments/tr_shape_dtlz2_100d/RESULTS.md`)
— that effective dimension relative to budget, not nominal dimension, governs
the benefit — rests entirely on a synthetic construction we built ourselves.
The most obvious reviewer objection is "does this generalize to a real
problem?" Candidates, ranked by how directly they'd answer that:

1. **LassoBench synthetic variants** (Šehić et al., AutoML 2022,
   arXiv:2111.02790; already in the benchmark table below as a
   single-objective scalability stress test) — tunable *true* sparsity of
   the regression coefficients maps directly to effective dimension, the
   closest available bridge between our synthetic SparseDTLZ2 and a real
   problem. Currently single-objective (test MSE); a natural second
   objective (number of active/nonzero coefficients, already computed
   internally) would make it bi-objective for free.
2. **Human-Powered Aircraft design family** (Namura & Takemori 2024,
   d=17/32/32/108) — real engineering, moderate-high dimension, currently
   single-objective but naturally has competing objectives (weight vs.
   drag vs. structural margin).
3. **PMO (Practical Molecular Optimization) latent-space tasks**
   (González-Duque et al. 2024, arXiv:2406.04739) — 128D latent-space
   molecule design with known low effective dimensionality relative to the
   much higher raw representation; drug-design objectives are naturally
   multi-objective (potency/synthesizability/toxicity).
4. **MOPTA08** (d=124, widely used single-objective constrained benchmark)
   — could be made bi-objective (mass vs. worst constraint violation
   magnitude), used by the REI paper above among others.
5. **YAHPO Gym / JAHS-Bench-201** — surrogate-backed multi-objective HPO/NAS
   benchmarks, cheap to run at scale (no live simulator), good for a
   large-batch high-nominal-dim study once the above are validated.

### Ranked recommendations (from this pass)

1. Validate the effective-dimension finding on a real problem — LassoBench
   (made bi-objective) or the HPA family are the best bridges available.
2. Upgrade `mab_shape` to a contextual bandit using an online effective-dim
   estimate (top-k PCA eigenvalue mass ratio) as context — cheap (the PCA
   computation already exists), and turns an empirical finding into a
   mechanism.
3. Add an explicit LABCAT/CMA-BO differentiation paragraph to `methods.tex`,
   plus a Hvarfner-style global vanilla-BO baseline (no trust region at
   all) and an AdaScale-TuRBO follow-up on the `ard_box` fix question.

---

## Benchmark / problem candidates

| Problem | d / M | Access | Matched baselines | Notes |
|---|---|---|---|---|
| **Penicillin** (fed-batch fermentation, ODE-based) | d=7, M=3 (yield/CO2/time) | `botorch.test_functions.multi_objective.Penicillin` — zero new deps | qLogEHVI, NSGA-II/III, MOEA/D, GDE3, global-LLM (MOHOLLM) | Canonical TR-BO benchmark ([TuRBO-Penicillin](https://github.com/HarryQL/TuRBO-Penicillin), Liang & Lai, NeurIPS'21 workshop). **Not differentiable** — hand-rolled Euler-loop ODE integrator, ~2500-step Python loop, no autograd. Rules out gradient-enhanced TR methods (LAGO). Cheapest way to test whether our Embedding BO/BOCF and LLM-subset ideas generalize past mixbox. |
| **VehicleSafety** (frontal crash design) | d=5, M=3 | `botorch.test_functions.multi_objective.VehicleSafety` | same as Penicillin (MOHOLLM) | Objectives: mass, collision acceleration, toe-board intrusion. Liang & Lai 2021 (same authors as TuRBO-Penicillin). |
| **CarSideImpact** (side-impact crash) | d=7, M=4 | `botorch.test_functions.multi_objective.CarSideImpact`, also Ax | same as Penicillin (MOHOLLM) | Highest objective-count real-world benchmark in this set. Tanabe & Ishibuchi. |
| **Summit SnAr / Baumgartner Suzuki-coupling** (reaction yield) | small d, mechanistic sim | [Summit](https://github.com/sustainable-processes/summit), public | SOBO, TSEMO, SNOBFIT, Nelder-Mead (ships its own baseline suite) | Closest structural match to our own color-mixing project — genuinely composite/cost-tradeoff, physically grounded like `mixbox`. |
| **Rover trajectory planning** (B-spline path) | d=60, M=2 (reward, distance-to-target) | Wang et al. 2018, widely reimplemented (e.g. [fast-cma-es](https://github.com/dietmarwo/fast-cma-es/blob/master/tutorials/RobotRover.adoc)) | MORBO, qNEHVI, qParEGO, TS-TCH, TSEMO, DGEMO, MOEA/D-EGO, LaMOO, NSGA-II, Sobol | MORBO's own real-world benchmark — even qNEHVI doesn't beat NSGA-II here, only MORBO does. Ships pre-wired in `facebookresearch/morbo`. |
| **Welded beam design** | d=4, V=4 constraints | classic engineering benchmark, public | same MORBO baseline set | Small-scale sanity check in MORBO's appendix. |
| **DTLZ3 / DTLZ5 / DTLZ7** | synthetic, 2 or 4 objectives | BoTorch/pymoo, trivial | same MORBO baseline set | Standard synthetic MOO suite. |
| **BOEngineeringBenchmark** (15 constrained problems, Ackley2D-style) | low-dim, `(objective, constraint)` pairs | [GitHub](https://github.com/rosenyu304/BOEngineeringBenchmark), public | GP/PFN × penalty/CEI/CEI+ | About feasibility constraints, not cost tradeoffs — different axis than most of the above. |
| **LassoBench** (weighted Lasso HPO) | tunable, up to 1000+D, single-objective | [github.com/ksehic/LassoBench](https://github.com/ksehic/LassoBench), pip-installable | — | Pure high-dim scalability stress test — deliberately *not* multi-objective, good control to isolate "does this scale" from "does it handle multiple objectives." Each eval takes seconds. Šehić et al., AutoML 2022. |
| **MOBO_aircraft** (CFRP wing aerostructural design) | d=5, M=2 (drag, weight) | not open-source (paper only) | NSGA-II, EPBII | New real-world domain (aerospace) not covered elsewhere on this list. |
| Optical AR/VR display design | d=146 | **not runnable** — hours per proprietary physics sim | — | MORBO's benchmark; aspirational only. |
| Mazda 3-vehicle design | d=222, V=54 | **not runnable** — original solve took ~3,000 CPU-years | — | MORBO's benchmark; aspirational only. |
| **LassoBench, made bi-objective** (MSE + active-coefficient count) | tunable true sparsity → tunable *effective dim*, up to 1000+D | same LassoBench install, +1 line to expose the existing internal coefficient count as a 2nd objective | — | Best available real bridge between synthetic `SparseDTLZ2` and a real problem for the tr_shape work — see "Follow-up review" section above. |
| **Human-Powered Aircraft design family** | d=17/32/32/108, real engineering, currently single-objective | Namura & Takemori 2024 (arXiv:2412.11456) | REI | Weight/drag/structural-margin are naturally competing objectives; not yet exposed as MO by the source paper. |
| **PMO molecular latent-space tasks** | 128D latent, known low effective dim | González-Duque et al. 2024 (arXiv:2406.04739) | — | Drug-design objectives (potency/synthesizability/toxicity) are naturally MO. |

**Recommendation**: Penicillin first (cheapest, validates pipeline generalizes
with near-zero engineering cost), then Summit's SnAr/Suzuki as the "real second
problem" (mirrors our composite/cost-tradeoff structure, ships its own baselines).
Rover is the best fit if/when we want a genuinely high-dimensional +
multi-objective test case for MORBO-based work specifically. **For the
tr_shape/effective-dimension line of work specifically**, LassoBench (made
bi-objective) is the top pick — see "Follow-up review" section above.

---

## Key technical clarifications (Q&A that came up in review)

- **TuRBO already runs multiple trust regions with their own GPs** — that part
  isn't new to MORBO. The actual differentiator is (1) shared data across TRs,
  (2) one shared hypervolume objective tying TRs together, (3) coordinated
  restart placement. "MORBO = TuRBO but TRs communicate" is close but
  underspecifies *what* they communicate about — it's coordination toward one
  shared goal, not just passing messages.
- **MORBO vs. TuRBO wasn't part of the main quantitative benchmark suite** — only
  a small illustrative comparison (their Fig. 2, DTLZ2 d=100 and MW7) to motivate
  the paper, not a rigorous head-to-head with the other 10 baselines.
- **Composite functions and hypervolume are orthogonal, not competing.**
  Hypervolume = how to score/rank candidates given objective values you already
  have. Composite = whether the GP models the objectives directly or a raw
  response that a known formula converts into them. You can combine both (that's
  the "MORBO of composite functions" gap) or pick composite+scalarization instead
  of hypervolume (that's what BoTier does).
- **MORBO needs no gradients of the true objective** — only for optimizing the
  (always-differentiable) acquisition surface built on the GP posterior, which is
  standard practice everywhere in BoTorch, not something specific to MORBO.
  Contrast directly with LAGO, which genuinely needs real gradient observations.
- **LEMOE ≈ MOHOLLM's core mechanism** (LLM fully replaces GP as both sampler and
  surrogate) applied to one specific domain (RISC-V) with a domain-specific
  warm-start, minus the hierarchical partitioning and minus the convergence proof.
- **AutoLead shares its LLM/BO scheduling idea with Trust-Aware BO** but isn't
  really multi-objective architecturally (single scalarized GP, not per-objective
  GPs + Expected Regret) — its real contribution is the SMILES/descriptor-space
  LLM-decoder trick, which is an unrelated problem.
- **BoTier explicitly avoids hypervolume** — EHVI is one of its baselines, not a
  component. Trades full-Pareto-front coverage for efficiently targeting one
  known-preference region.

---

## Tools & Libraries

### Platypus
[github.com/Project-Platypus/Platypus](https://github.com/Project-Platypus/Platypus)

A Python framework for evolutionary computing focused on multi-objective evolutionary algorithms (MOEAs) — an importable library, not a framework you extend.

Implements 10 MOEAs: **NSGA-II, NSGA-III, MOEA/D, IBEA, Epsilon-MOEA, SPEA2, GDE3, OMOPSO, SMPSO, Epsilon-NSGA-II**. Actively maintained (latest release Oct 2024, 654 stars).

Why it matters for this review: this covers most of the classical baselines that show up across nearly every paper above — NSGA-II/III appear in almost all of them, MOEA/D underlies MOEA/D-EGO in MORBO's baseline list, and GDE3 is one of MOHOLLM's own baselines directly. Rather than hand-implementing NSGA-II or GDE3 from scratch to compare against, this is a mature, correct, drop-in source for the "standard MOEA baseline set" across whichever benchmark problems we pick (Penicillin, VehicleSafety, etc.).

---

### DGEMO — Diversity-Guided Efficient Multi-Objective Optimization
[github.com/yunshengtian/DGEMO](https://github.com/yunshengtian/DGEMO) · Konaković Luković, Tian & Matusik · NeurIPS 2020

**This is the actual reference implementation of a baseline MORBO itself cites** — MORBO's related-work section describes DGEMO as using *"a hypervolume-based objective with heuristics to encourage diversity while exploring the PF."* If we want a real MORBO-vs-DGEMO comparison rather than approximating or citing secondhand numbers, this is the authentic source.

Architecture is modular — a multi-objective BO framework with swappable components:
- Surrogate: Gaussian process
- Acquisition: identity function (default), or PI/EI/UCB
- Multi-objective solver: ParetoDiscovery (default), or NSGA-II, MOEA/D
- Selection: diversity-guided criterion for picking the next batch

Tested on ZDT1-3, DTLZ1-6, OKA1-2, VLMOP2-3, and RE problems — good overlap with the synthetic suites MOHOLLM and MORBO both used. Usage: `python main.py --problem dtlz1 --n-var 6 --n-obj 2 --n-iter 20` for single runs, or `python scripts/run.py` for batch experiments across algorithm/problem combinations; custom problems follow templates in `problems/`. Moderate activity (114 stars, 25 forks; last meaningful update Feb 2023).

**Possible use beyond "just a baseline"**: the surrogate/acquisition/solver separation already built into this codebase could plausibly serve as a starting scaffold for "MORBO of composite functions" — swapping in a composite-response GP instead of modeling objectives directly is a smaller lift here than building from scratch, since that separation already exists.

---

## Reproducing/using our own project's related infra

This review grew out of `D:\SURP\bayesian-optimization-benchmarks` (color-mixing
BO benchmark, `embedding-bo` branch). Relevant reusable pieces if composite or
high-dim methods get implemented here:
- `src/residual_bo_solver.py` — `ResidualBOCFSolver` already does composite-GP +
  known-objective-inside-acquisition for a single region (the non-MORBO half of
  "MORBO of composite functions").
- `ExpandedTuRBOSolver` / `LLMAssistedTuRBOSolver` (in `expanded_solvers.py`,
  `origin/main` branch) — existing gradient-free trust-region baseline to build
  "LLM-assisted MORBO" or a naive scalarized-TuRBO strawman from.
