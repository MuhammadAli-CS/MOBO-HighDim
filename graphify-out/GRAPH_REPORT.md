# Graph Report - .  (2026-07-08)

## Corpus Check
- Corpus is ~34,696 words - fits in a single context window. You may not need a graph.

## Summary
- 355 nodes · 594 edges · 24 communities (21 shown, 3 thin omitted)
- Extraction: 96% EXTRACTED · 4% INFERRED · 0% AMBIGUOUS · INFERRED: 25 edges (avg confidence: 0.6)
- Token cost: 95,000 input · 7,000 output

## Community Hubs (Navigation)
- State Management (TRBOState)
- Trust Region Types
- Rover Problem
- Core Utilities & Base Types
- Candidate Generation & LLM Candidates
- Composite Problems & Replication
- BoTier LLM Solver
- Benchmark Function
- Benchmark Suite & Literature
- DTLZ2 100D Comparison Results
- LLM-Assisted MORBO & Trust Schedule
- Plotting & Visualization
- Penicillin & MOO Literature
- Composite BO / STCH Theory
- BoTier & Hierarchical Scalarization
- Governance Docs
- Composite MORBO & Correlation Ablation
- LLM-for-BO Literature Misc
- BO Engineering Benchmark
- Summit Suzuki Benchmark

## God Nodes (most connected - your core abstractions)
1. `TRBOState` - 34 edges
2. `TrustRegion` - 25 edges
3. `run_one_replication()` - 17 edges
4. `MORBO: Multi-Objective Bayesian Optimization over High-Dimensional Search Spaces (Daulton et al., UAI 2022)` - 16 edges
5. `TurboHParams` - 15 edges
6. `ScalarizedTrustRegion` - 15 edges
7. `TabuSet` - 13 edges
8. `HypervolumeTrustRegion` - 12 edges
9. `get_fitted_model()` - 12 edges
10. `BenchmarkFunction` - 11 edges

## Surprising Connections (you probably didn't know these)
- `Code of Conduct` --semantically_similar_to--> `Contributing to morbo`  [INFERRED] [semantically similar]
  CODE_OF_CONDUCT.md → CONTRIBUTING.md
- `Comparison chart: Composite MORBO vs MORBO vs TuRBO+Chebyshev (DTLZ2 d=100, seed 0)` --references--> `Composite MORBO (composite_morbo, DTLZ2 raw response modeling)`  [EXTRACTED]
  experiments/fig2_dtlz2_100d/comparison_seed0.png → README.md
- `Open gap: MORBO of composite functions` --rationale_for--> `Composite MORBO (composite_morbo, DTLZ2 raw response modeling)`  [INFERRED]
  LITERATURE_REVIEW.md → README.md
- `run_botier_bo()` --calls--> `get_fitted_model()`  [EXTRACTED]
  botier_llm/solver.py → morbo/utils.py
- `LLM candidate count decay schedule (reuses decay_function, not Bernoulli accept/reject)` --conceptually_related_to--> `Trust-Aware LLM-Assisted BO (Zhou, Wang, Gu & Tan, 2026)`  [EXTRACTED]
  README.md → LITERATURE_REVIEW.md

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **LLM-in-the-BO-loop family (varying replacement degree of the GP)** — literature_review_trust_aware_llm_assisted_bo, literature_review_llms_for_bo_scientific_domains, literature_review_mohollm, literature_review_lemoe, literature_review_autolead [INFERRED 0.85]
- **Composite-objective BO modeling pattern (GP on raw response + known reduction)** — literature_review_composite_bo_stch, literature_review_botier, readme_composite_morbo, readme_maddox_composite_bo_paper [INFERRED 0.85]
- **MORBO paper's main quantitative benchmark suite** — literature_review_morbo_paper, literature_review_rover_benchmark, literature_review_welded_beam_benchmark, literature_review_dtlz_suite, literature_review_optical_display_benchmark, literature_review_mazda_benchmark [EXTRACTED 1.00]

## Communities (24 total, 3 thin omitted)

### Community 0 - "State Management (TRBOState)"
Cohesion: 0.07
Nodes (26): Module, device, dtype, MCAcquisitionObjective, Model, Tensor, Retrieve current tabu points, Log a BO iteration, decrement tenures, and prune set. (+18 more)

### Community 1 - "Trust Region Types"
Cohesion: 0.07
Nodes (24): MCSampler, HypervolumeTrustRegion, Any, MCAcquisitionObjective, Tensor, r"""Construct a TurboHParams object from a dict.          This automatically f, r"""A trust region object.      This is a variation of the TuRBO algorithm pre, r"""Return True if the model training data was updated. (+16 more)

### Community 2 - "Rover Problem"
Cohesion: 0.06
Nodes (15): AABoxes, AdditiveCosts, ConstantOffsetFn, ConstCost, ConstObstacleCost, create_cost_large(), create_large_domain(), get_rover_fn() (+7 more)

### Community 3 - "Core Utilities & Base Types"
Cohesion: 0.08
Nodes (33): ABC, Enum, InputTransform, # TODO: re-evaluate/improve this., # TODO: evaluate more principled strategies for HV TR center, # NOTE: this currently shares data across trust regions and restarts., # NOTE: This currently shares data across trust regions and restarts., # NOTE: Always apply the decay function even if we aren't restarting. This may h (+25 more)

### Community 4 - "Candidate Generation & LLM Candidates"
Cohesion: 0.08
Nodes (32): BaseModel, ObjectiveTier, BoxDecomposition, CandidateSelectionOutput, get_partitioning(), _make_unstandardizer(), preds_and_feas(), Tensor (+24 more)

### Community 5 - "Composite Problems & Replication"
Cohesion: 0.10
Nodes (27): fetch_data(), Any, composite_dtlz2_reduction(), composite_dtlz2_curve_reduction(), get_composite_dtlz2_curve_fn(), callable, Tensor, r"""Construct the raw-response (curve) function and bounds.      Args:         d (+19 more)

### Community 6 - "BoTier LLM Solver"
Cohesion: 0.22
Nodes (13): propose_tiers(), Anthropic, Tensor, r"""Propose a BoTier priority ordering and per-objective thresholds.      Args:, TierProposal, hierarchical_scalarization(), percentile_thresholds(), Tensor (+5 more)

### Community 7 - "Benchmark Function"
Cohesion: 0.16
Nodes (8): BenchmarkFunction, device, dtype, Tensor, r"""Loop over observations one by one and record the resulting HVs., r"""This is a wrapper that wraps the function callable and implements additional, r"""Evaluate the function and return the possibly noisy observations. Also, r"""Records the current noise-free PF and HV.

### Community 8 - "Benchmark Suite & Literature"
Cohesion: 0.20
Nodes (11): DGEMO: Diversity-Guided Efficient Multi-Objective Optimization (NeurIPS 2020), DTLZ3/5/7 synthetic benchmark suite, LAGO: Combining Trust Region Methods and Bayesian Optimization (2026), LassoBench (weighted Lasso HPO, up to 1000+D single-objective), Mazda 3-vehicle design (d=222, not runnable), MORBO: Multi-Objective Bayesian Optimization over High-Dimensional Search Spaces (Daulton et al., UAI 2022), Optical AR/VR display design (d=146, not runnable), Rover trajectory planning benchmark (d=60, M=2) (+3 more)

### Community 9 - "DTLZ2 100D Comparison Results"
Cohesion: 0.31
Nodes (9): Comparison chart: Composite MORBO vs MORBO vs TuRBO+Chebyshev (DTLZ2 d=100, seed 0), Collaborative (vs. independent) batch selection, Data sharing (track_history) as enabling infrastructure, MORBO vs. Scalarized-TuRBO on DTLZ2 (d=100) — Results, MORBO run (fig2_dtlz2_100d, seed 0, HV=20.02), TR center reselection tied to shared global frontier, TuRBO + Chebyshev scalarizations run (fig2_dtlz2_100d, seed 0, HV=16.92), ExpandedTuRBOSolver / LLMAssistedTuRBOSolver (expanded_solvers.py, sibling project) (+1 more)

### Community 10 - "LLM-Assisted MORBO & Trust Schedule"
Cohesion: 0.22
Nodes (9): AutoLead: LLM-Guided BO for Multi-Objective Lead Optimization (2025), Expected Regret (ER) acquisition, LLM-as-inverse-decoder trick (descriptor vector -> SMILES), Team next steps (Ricky/Sahas/Muhammad directions), Trust-Aware LLM-Assisted BO (Zhou, Wang, Gu & Tan, 2026), Trust probability schedule p_t = lambda/t^2, Two candidate-generation bugs fixed in morbo/gen.py, LLM-assisted MORBO (llm_morbo, sparse perturbation proposals) (+1 more)

### Community 11 - "Plotting & Visualization"
Cohesion: 0.25
Nodes (7): discover_labels(), hv_trace(), objective_Y(), Tensor, r"""Objective-space observations for a saved run.      Prefers `objective_histor, Find every label with a saved `<seed>_<label>.pt` file under exp_dir., Running hypervolume of Y[:k] for k = 1..n (every `step` evals).

### Community 12 - "Penicillin & MOO Literature"
Cohesion: 0.25
Nodes (8): Adaptive KD-tree space partitioning, CarSideImpact benchmark (d=7, M=4), LEMOE: LLM-Enhanced MOBO for Microarchitecture Exploration (DAC 2025), MOHOLLM: Multi-Objective Hierarchical Optimization with LLMs (2026), Penicillin fed-batch fermentation benchmark (d=7, M=3), Platypus (MOEA library: NSGA-II/III, MOEA/D, GDE3, etc.), VehicleSafety frontal crash benchmark (d=5, M=3), Composite MORBO on Penicillin simulator (checkpointed Euler-integrator state)

### Community 13 - "Composite BO / STCH Theory"
Cohesion: 0.40
Nodes (6): Composite BO for Multi-Objective Problems with Smooth Tchebycheff Scalarisation (2026), Composite objective structure J(theta) = g(L(theta, Y(theta))), MOBO of Composite Aircraft Wings Using Various Carbon Fibers (2026), Open gap: MORBO of composite functions, ResidualBOCFSolver (src/residual_bo_solver.py, sibling project), STCH: Smooth Tchebycheff Scalarization (Lin & Zhang, arXiv:2402.19078)

### Community 14 - "BoTier & Hierarchical Scalarization"
Cohesion: 0.50
Nodes (5): BoTier: MOBO with Tiered Composite Objectives (2025), Chimera (Aspuru-Guzik group, prior scalarization method), Hierarchical scalarization Xi = sum(min(psi_i,t_i) * prod H(psi_j - t_j)), Sequential greedy hypervolume improvement (HVI) batch selection, LLM-automated BoTier (botier_llm/, standalone tiered-utility BO)

### Community 15 - "Governance Docs"
Cohesion: 0.50
Nodes (4): Contributor Covenant v1.4, Code of Conduct, Contributor License Agreement (CLA), Contributing to morbo

### Community 16 - "Composite MORBO & Correlation Ablation"
Cohesion: 0.67
Nodes (4): Composite MORBO (composite_morbo, DTLZ2 raw response modeling), Correlation ablation: DTLZ2 curve response (composite_dtlz2_curve.py), Kronecker-structured multi-task GP composite (KroneckerMultiTaskGP), Maddox, Feng & Balandat — Optimizing High-Dimensional Physics Simulations via Composite BO (NeurIPS 2021 ML4PS)

## Ambiguous Edges - Review These
- `Composite objective structure J(theta) = g(L(theta, Y(theta)))` → `MOBO of Composite Aircraft Wings Using Various Carbon Fibers (2026)`  [AMBIGUOUS]
  LITERATURE_REVIEW.md · relation: conceptually_related_to

## Knowledge Gaps
- **25 isolated node(s):** `Contributor Covenant v1.4`, `Contributor License Agreement (CLA)`, `Expected Regret (ER) acquisition`, `LLMs for BO in Scientific Domains: Are We There Yet?`, `LLMNN (LLM cluster centers + nearest-neighbor, no BO)` (+20 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **3 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What is the exact relationship between `Composite objective structure J(theta) = g(L(theta, Y(theta)))` and `MOBO of Composite Aircraft Wings Using Various Carbon Fibers (2026)`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **Why does `TRBOState` connect `State Management (TRBOState)` to `Trust Region Types`, `Core Utilities & Base Types`, `Candidate Generation & LLM Candidates`, `Composite Problems & Replication`?**
  _High betweenness centrality (0.172) - this node is a cross-community bridge._
- **Why does `TrustRegion` connect `Trust Region Types` to `State Management (TRBOState)`, `Core Utilities & Base Types`?**
  _High betweenness centrality (0.078) - this node is a cross-community bridge._
- **Why does `BenchmarkFunction` connect `Benchmark Function` to `State Management (TRBOState)`, `Composite Problems & Replication`?**
  _High betweenness centrality (0.058) - this node is a cross-community bridge._
- **Are the 5 inferred relationships involving `TRBOState` (e.g. with `CandidateSelectionOutput` and `HypervolumeTrustRegion`) actually correct?**
  _`TRBOState` has 5 INFERRED edges - model-reasoned connections that need verification._
- **Are the 3 inferred relationships involving `TrustRegion` (e.g. with `TabuSet` and `TRBOState`) actually correct?**
  _`TrustRegion` has 3 INFERRED edges - model-reasoned connections that need verification._
- **Are the 3 inferred relationships involving `run_one_replication()` (e.g. with `composite_dtlz2_reduction()` and `composite_dtlz2_curve_reduction()`) actually correct?**
  _`run_one_replication()` has 3 INFERRED edges - model-reasoned connections that need verification._