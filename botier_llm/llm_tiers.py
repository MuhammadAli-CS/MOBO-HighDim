#!/usr/bin/env python3
r"""One-shot LLM proposal of BoTier tier ordering + per-objective thresholds.

BoTier itself takes the priority ordering and every threshold `t_i` as a
static, hand-specified input. This module automates that one input: given a
natural-language description of the problem and (optionally) a small
warm-start sample of achievable objective ranges, an LLM proposes a priority
ordering and a threshold per objective, once per run -- not a per-iteration
hook, so cost is a single call.

Uses the official `anthropic` SDK with structured output
(`client.messages.parse`) rather than free-text parsing. Reads
`ANTHROPIC_API_KEY` from the environment via the SDK's standard credential
resolution.
"""
from typing import List, Optional

import torch
from anthropic import Anthropic
from pydantic import BaseModel, Field
from torch import Tensor

MODEL = "claude-opus-4-8"


class ObjectiveTier(BaseModel):
    name: str = Field(description="Objective name, matching one of the given names exactly.")
    threshold: float = Field(
        description="The value (maximization convention -- higher is better) beyond "
        "which this objective is 'good enough' and lower-priority objectives should "
        "start being considered."
    )


class TierProposal(BaseModel):
    # A list of objects, in priority order (index 0 = highest priority), rather
    # than a free-form dict -- dict-valued fields don't play well with
    # structured-output schemas (they can't set `additionalProperties: false`).
    tiers: List[ObjectiveTier] = Field(
        description="One entry per objective, ordered from highest to lowest "
        "priority. Must contain exactly one entry per given objective name."
    )
    rationale: str = Field(
        description="One or two sentences on why this ordering and these thresholds "
        "were chosen."
    )


def propose_tiers(
    problem_description: str,
    objective_names: List[str],
    warm_start_Y: Optional[Tensor] = None,
    client: Optional[Anthropic] = None,
):
    r"""Propose a BoTier priority ordering and per-objective thresholds.

    Args:
        problem_description: natural-language description of the problem --
            what the objectives are, their units, what "good" looks like.
        objective_names: names of the objectives, in the same column order
            as the optimization's own `Y` tensor (return values are mapped
            back to this order).
        warm_start_Y: optional `n x M`-dim tensor of warm-start (e.g. Sobol)
            objective observations, maximization convention, so the LLM sees
            actual achievable ranges instead of guessing blind.
        client: an `anthropic.Anthropic` client; constructed from the
            environment (`ANTHROPIC_API_KEY`) if not provided.

    Returns:
        A tuple `(order, thresholds, rationale)`:
            order: `M`-dim `LongTensor`, a permutation of `range(M)` mapping
                priority rank -> index into `objective_names`/`Y` columns.
            thresholds: `M`-dim tensor, `thresholds[k]` is the threshold for
                objective `order[k]` (matching `hierarchical_scalarization`'s
                expected layout in `botier_llm/solver.py`).
            rationale: the LLM's stated rationale, for logging.
    """
    client = client or Anthropic()

    ranges_text = ""
    if warm_start_Y is not None:
        mins = warm_start_Y.min(dim=0).values.tolist()
        maxs = warm_start_Y.max(dim=0).values.tolist()
        medians = warm_start_Y.median(dim=0).values.tolist()
        lines = [
            f"  - {name}: observed range [{lo:.4g}, {hi:.4g}], median {med:.4g}"
            for name, lo, hi, med in zip(objective_names, mins, maxs, medians)
        ]
        ranges_text = (
            "\nWarm-start observations (all in a maximization convention -- "
            "higher is always better) from a small random sample of the "
            "search space:\n" + "\n".join(lines)
        )

    prompt = (
        f"{problem_description}\n\n"
        f"Objectives (all maximized, higher is better): {', '.join(objective_names)}."
        f"{ranges_text}\n\n"
        "Propose a priority ordering over these objectives and a threshold for "
        "each, for a tiered/hierarchical optimization: the optimizer will fully "
        "satisfy the highest-priority objective's threshold before letting the "
        "next-priority objective influence the search at all. Choose thresholds "
        "that represent 'good enough, not maximal' for each objective given the "
        "observed ranges -- setting a threshold at the observed maximum leaves no "
        "room for lower-priority objectives to matter."
    )

    response = client.messages.parse(
        model=MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
        output_format=TierProposal,
    )
    proposal = response.parsed_output
    proposed_names = [tier.name for tier in proposal.tiers]

    if set(proposed_names) != set(objective_names):
        raise ValueError(
            f"LLM proposed objective names {proposed_names} that don't match "
            f"the given {objective_names}."
        )

    name_to_idx = {name: i for i, name in enumerate(objective_names)}
    order = torch.tensor(
        [name_to_idx[tier.name] for tier in proposal.tiers], dtype=torch.long
    )
    thresholds = torch.tensor(
        [tier.threshold for tier in proposal.tiers], dtype=torch.double
    )
    return order, thresholds, proposal.rationale
