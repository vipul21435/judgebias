"""Synthetic judges with known biases, for tests and examples.

``simulate`` draws prompts with a hidden "true" better answer, then a judge
whose verdict is pulled by three knobs: ``position`` (extra probability of
picking the first-shown answer), ``length`` (extra pull per unit of log length
ratio) and ``self_pref`` (extra pull toward answers from the judge's family).
Each pair is judged in both orders, so flip rate and swap-corrected measures
have data to work with. Human labels follow the true answer with a fixed
accuracy. Nothing here calls a network.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .data import Judgment

MODELS = ("alpha-7b", "beta-13b", "gamma-70b", "delta-8x7b")


def simulate(
    n_pairs: int = 200,
    *,
    position: float = 0.0,
    length: float = 0.0,
    self_pref: float = 0.0,
    judge: str = "gamma-70b",
    noise: float = 0.15,
    tie_rate: float = 0.05,
    human_accuracy: float = 0.85,
    repeats: int = 1,
    seed: int = 0,
) -> list[Judgment]:
    """Return judgments from a synthetic judge with the given biases."""
    rng = np.random.default_rng(seed)
    out: list[Judgment] = []
    for i in range(n_pairs):
        pair_id = f"p{i:05d}"
        model_a, model_b = rng.choice(MODELS, size=2, replace=False)
        len_a = int(rng.lognormal(mean=5.5, sigma=0.5))
        len_b = int(rng.lognormal(mean=5.5, sigma=0.5))
        true = "A" if rng.random() < 0.5 else "B"
        human = true if rng.random() < human_accuracy else ("B" if true == "A" else "A")
        if rng.random() < tie_rate:
            human = "tie"
        log_ratio = float(np.log((len_a + 1) / (len_b + 1)))
        fam = judge.split("-")[0]
        own_pull = 0.0
        if model_a.startswith(fam) != model_b.startswith(fam):
            own_pull = self_pref if model_a.startswith(fam) else -self_pref
        for order in ("AB", "BA"):
            for rep in range(repeats):
                # score > 0 favours A
                score = (1.0 if true == "A" else -1.0) * (1.0 - noise)
                score += length * log_ratio
                score += own_pull
                score += position if order == "AB" else -position
                score += rng.normal(0.0, noise)
                if abs(score) < tie_rate / 2:
                    verdict = "tie"
                else:
                    verdict = "A" if score > 0 else "B"
                out.append(
                    Judgment(
                        pair_id=pair_id,
                        verdict=verdict,
                        order=order,
                        judge=judge,
                        len_a=len_a,
                        len_b=len_b,
                        model_a=str(model_a),
                        model_b=str(model_b),
                        human=human,
                        repeat=rep,
                    )
                )
    return out


def to_rows(judgments: list[Judgment]) -> list[dict[str, Any]]:
    rows = []
    for j in judgments:
        rows.append(
            {
                "pair_id": j.pair_id,
                "verdict": j.verdict,
                "order": j.order,
                "judge": j.judge,
                "len_a": j.len_a,
                "len_b": j.len_b,
                "model_a": j.model_a,
                "model_b": j.model_b,
                "human": j.human,
                "repeat": j.repeat,
            }
        )
        if j.flipped is not None:
            rows[-1]["flipped"] = j.flipped
    return rows
