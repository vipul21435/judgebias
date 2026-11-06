"""The audit measures.

Every measure takes a list of :class:`Judgment` and returns a plain dict of
numbers, with bootstrap intervals where a point estimate alone would mislead.
Bootstrap units are prompts (``pair_id``), so the two presentation orders of a
pair, and repeated samples of it, are resampled together.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from typing import Any

import numpy as np

from .data import Judgment
from .stats import Estimate, bootstrap, cohen_kappa, logistic_fit


def _by_pair(judgments: Sequence[Judgment]) -> list[list[Judgment]]:
    groups: dict[str, list[Judgment]] = defaultdict(list)
    for j in judgments:
        groups[j.pair_id].append(j)
    return list(groups.values())


def distinct(judgments: Sequence[Judgment]) -> list[Judgment]:
    """Keep the first row of each (pair, order, judge, repeat).

    A file that lists one judge verdict once per human label repeats the same
    verdict several times; the position and length measures describe the
    judge, so they count each verdict once. The agreement and self-preference
    measures compare against human labels row by row and use every row.
    """
    seen: set[tuple[str, str, str, int]] = set()
    out: list[Judgment] = []
    for j in judgments:
        key = (j.pair_id, j.order, j.judge, j.repeat)
        if key not in seen:
            seen.add(key)
            out.append(j)
    return out


def _flat(groups: Sequence[Sequence[Judgment]]) -> list[Judgment]:
    return [j for g in groups for j in g]


def _nan_if_empty(num: float, den: float) -> float:
    return float(num / den) if den else float("nan")


# ---------------------------------------------------------------------------
# position bias


def _first_win_rate(groups: Sequence[Sequence[Judgment]]) -> float:
    picks = [j.picked_first for j in _flat(groups)]
    decided = [p for p in picks if p is not None]
    return _nan_if_empty(sum(decided), len(decided))


def _flip_rate(groups: Sequence[Sequence[Judgment]]) -> float:
    """Share of pairs judged in both orders whose verdict changed with the order."""
    flips = 0
    both = 0
    for g in groups:
        ab = [j.verdict for j in g if j.order == "AB"]
        ba = [j.verdict for j in g if j.order == "BA"]
        if not ab or not ba:
            continue
        both += 1
        if ab[0] != ba[0]:
            flips += 1
    return _nan_if_empty(flips, both)


def _consistent_first_rate(groups: Sequence[Sequence[Judgment]]) -> float:
    """Among flipped pairs, the share where the judge took the first answer both times."""
    first_both = 0
    flipped = 0
    for g in groups:
        ab = [j for j in g if j.order == "AB"]
        ba = [j for j in g if j.order == "BA"]
        if not ab or not ba or ab[0].verdict == ba[0].verdict:
            continue
        flipped += 1
        if ab[0].picked_first and ba[0].picked_first:
            first_both += 1
    return _nan_if_empty(first_both, flipped)


def position_bias(
    judgments: Sequence[Judgment], *, reps: int = 1000, seed: int = 0
) -> dict[str, Any]:
    """First-position win rate and flip rate under swapped presentation.

    ``first_win_rate`` is the share of decided verdicts that picked whichever
    answer was shown first; 0.5 is unbiased. ``flip_rate`` needs pairs judged
    in both orders: the share whose verdict changed when the order changed.
    ``first_both_rate`` is, among flipped pairs, the share where the judge
    chose the first-shown answer both times (a pure position preference) as
    opposed to the second both times. Copied rows count once (see
    :func:`distinct`).
    """
    n_rows = len(judgments)
    judgments = distinct(judgments)
    groups = _by_pair(judgments)
    picks = [j.picked_first for j in judgments]
    both = sum(
        1
        for g in groups
        if any(j.order == "AB" for j in g) and any(j.order == "BA" for j in g)
    )
    return {
        "n_rows": n_rows,
        "n_judgments": len(judgments),
        "n_pairs": len(groups),
        "n_decided": sum(p is not None for p in picks),
        "n_ties": sum(p is None for p in picks),
        "first_win_rate": bootstrap(groups, _first_win_rate, reps=reps, seed=seed).as_dict(),
        "n_pairs_both_orders": both,
        "flip_rate": bootstrap(groups, _flip_rate, reps=reps, seed=seed).as_dict(),
        "first_both_rate": bootstrap(groups, _consistent_first_rate, reps=reps, seed=seed)
        .as_dict(),
    }


# ---------------------------------------------------------------------------
# length bias


def _length_rows(judgments: Sequence[Judgment]) -> tuple[np.ndarray, np.ndarray]:
    xs: list[float] = []
    ys: list[float] = []
    for j in judgments:
        if j.verdict == "tie" or j.len_a is None or j.len_b is None:
            continue
        # log length ratio of A over B, verdict 1 if A won
        xs.append(float(np.log((j.len_a + 1) / (j.len_b + 1))))
        ys.append(1.0 if j.verdict == "A" else 0.0)
    return np.asarray(xs), np.asarray(ys)


def _slope(groups: Sequence[Sequence[Judgment]]) -> float:
    x, y = _length_rows(_flat(groups))
    if x.size < 2:
        return float("nan")
    return logistic_fit(x, y)[1]


def _longer_win_rate(groups: Sequence[Sequence[Judgment]]) -> float:
    wins = 0
    decided = 0
    for j in _flat(groups):
        if j.verdict == "tie" or j.len_a is None or j.len_b is None or j.len_a == j.len_b:
            continue
        decided += 1
        longer = "A" if j.len_a > j.len_b else "B"
        wins += j.verdict == longer
    return _nan_if_empty(wins, decided)


def length_bias(
    judgments: Sequence[Judgment], *, reps: int = 1000, seed: int = 0
) -> dict[str, Any]:
    """Verdict against length difference.

    ``longer_win_rate`` is the share of decided verdicts with unequal lengths
    that went to the longer answer. ``slope`` is the coefficient of
    ``log((len_a+1)/(len_b+1))`` in a logistic fit of P(A wins); 0 means
    length does not move the verdict, positive means longer answers win more.
    Both come with a bootstrap interval over prompts. Copied rows count once
    (see :func:`distinct`).
    """
    judgments = distinct(judgments)
    groups = _by_pair(judgments)
    x, y = _length_rows(judgments)
    intercept, slope = logistic_fit(x, y) if x.size >= 2 else (float("nan"), float("nan"))
    return {
        "n_used": int(x.size),
        "longer_win_rate": bootstrap(groups, _longer_win_rate, reps=reps, seed=seed).as_dict(),
        "intercept": intercept,
        "slope": Estimate(
            slope,
            *_interval(groups, _slope, reps, seed),
            n=len(groups),
        ).as_dict(),
    }


def _interval(groups, statistic, reps, seed) -> tuple[float, float]:
    est = bootstrap(groups, statistic, reps=reps, seed=seed)
    return est.low, est.high


# ---------------------------------------------------------------------------
# self-preference


def self_preference(
    judgments: Sequence[Judgment], *, reps: int = 1000, seed: int = 0
) -> dict[str, Any]:
    """Win rate of answers from the judge's own family against the rest.

    A judgment counts when exactly one of ``model_a`` / ``model_b`` matches the
    judge's family, where family is the model name up to the first ``-`` or
    ``/`` (``judge="gpt-4"`` and ``model_a="gpt-3.5"`` share family ``gpt``).
    ``own_win_rate`` above the judge's overall win rate for that family on
    human labels suggests self-preference; the human comparison is only
    reported when human labels exist.
    """
    groups = _by_pair(judgments)

    def own(g: Sequence[Sequence[Judgment]], use_human: bool) -> float:
        wins = 0
        decided = 0
        for j in _flat(g):
            if j.model_a is None or j.model_b is None:
                continue
            fam = family(j.judge)
            a_own, b_own = family(j.model_a) == fam, family(j.model_b) == fam
            if a_own == b_own:
                continue
            label = j.human if use_human else j.verdict
            if label is None or label == "tie":
                continue
            decided += 1
            wins += label == ("A" if a_own else "B")
        return _nan_if_empty(wins, decided)

    n_used = 0
    for j in judgments:
        if j.model_a is None or j.model_b is None:
            continue
        fam = family(j.judge)
        if (family(j.model_a) == fam) != (family(j.model_b) == fam):
            n_used += 1
    out: dict[str, Any] = {
        "judge_family": family(judgments[0].judge) if judgments else None,
        "n_used": n_used,
        "own_win_rate": bootstrap(
            groups, lambda g: own(g, False), reps=reps, seed=seed
        ).as_dict(),
    }
    if any(j.human is not None for j in judgments):
        out["own_win_rate_human"] = bootstrap(
            groups, lambda g: own(g, True), reps=reps, seed=seed
        ).as_dict()
        out["excess"] = out["own_win_rate"]["value"] - out["own_win_rate_human"]["value"]
    return out


def family(model: str | None) -> str | None:
    if model is None:
        return None
    text = model.lower()
    for sep in ("/", "-", "_", ":"):
        text = text.split(sep, 1)[0]
    return text


# ---------------------------------------------------------------------------
# agreement with humans


def _labelled(groups: Sequence[Sequence[Judgment]], ties: str) -> tuple[list[str], list[str]]:
    a: list[str] = []
    b: list[str] = []
    for j in _flat(groups):
        if j.human is None:
            continue
        if ties == "drop" and (j.human == "tie" or j.verdict == "tie"):
            continue
        a.append(j.verdict)
        b.append(j.human)
    return a, b


def agreement(
    judgments: Sequence[Judgment],
    *,
    ties: str = "keep",
    reps: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    """Raw agreement and Cohen's kappa between the judge and human labels.

    ``ties="keep"`` treats tie as a third label on both sides; ``ties="drop"``
    removes any row where either side said tie. Both are reported with
    bootstrap intervals over prompts.
    """
    if ties not in ("keep", "drop"):
        raise ValueError("ties must be 'keep' or 'drop'")
    groups = _by_pair(judgments)

    def raw(g: Sequence[Sequence[Judgment]]) -> float:
        a, b = _labelled(g, ties)
        return _nan_if_empty(sum(x == y for x, y in zip(a, b, strict=True)), len(a))

    def kappa(g: Sequence[Sequence[Judgment]]) -> float:
        a, b = _labelled(g, ties)
        return cohen_kappa(a, b)

    a, b = _labelled(groups, ties)
    confusion = Counter(zip(a, b, strict=True))
    return {
        "ties": ties,
        "n_labelled": len(a),
        "n_human_ties": sum(1 for j in judgments if j.human == "tie"),
        "n_judge_ties": sum(1 for j in judgments if j.human is not None and j.verdict == "tie"),
        "raw": bootstrap(groups, raw, reps=reps, seed=seed).as_dict(),
        "kappa": bootstrap(groups, kappa, reps=reps, seed=seed).as_dict(),
        "confusion": {f"judge={x},human={y}": c for (x, y), c in sorted(confusion.items())},
    }


# ---------------------------------------------------------------------------
# repeat consistency


def repeat_consistency(judgments: Sequence[Judgment]) -> dict[str, Any]:
    """How often repeated judgments of the same pair in the same order agree.

    A group is one (pair, order, judge); its samples are told apart by the
    ``repeat`` index, and rows that share all four keys are copies of one
    sample (for example one judge verdict listed once per human label) and
    count once. Different judges on the same pair are not repeats of each
    other; their agreement is what :func:`agreement` measures.

    ``unanimous_rate`` is the share of groups with 2+ samples whose verdicts
    all match; ``pairwise_agreement`` is the share of all sample pairs within
    a group that match.
    """
    samples: dict[tuple[str, str, str], dict[int, str]] = defaultdict(dict)
    for j in judgments:
        samples[(j.pair_id, j.order, j.judge)].setdefault(j.repeat, j.verdict)
    multi = [list(v.values()) for v in samples.values() if len(v) >= 2]
    unanimous = sum(len(set(v)) == 1 for v in multi)
    agree = 0
    total = 0
    for v in multi:
        for i in range(len(v)):
            for k in range(i + 1, len(v)):
                total += 1
                agree += v[i] == v[k]
    return {
        "n_groups": len(multi),
        "unanimous_rate": _nan_if_empty(unanimous, len(multi)),
        "pairwise_agreement": _nan_if_empty(agree, total),
    }


# ---------------------------------------------------------------------------


def audit(
    judgments: Sequence[Judgment],
    *,
    ties: str = "keep",
    reps: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    """Run every measure that the data supports.

    Self-preference is reported only when some judgment pits the judge's own
    family against another; it is omitted otherwise.
    """
    out: dict[str, Any] = {
        "n_judgments": len(judgments),
        "judges": sorted({j.judge for j in judgments}),
        "position": position_bias(judgments, reps=reps, seed=seed),
        "repeat": repeat_consistency(judgments),
    }
    if any(j.len_a is not None and j.len_b is not None for j in judgments):
        out["length"] = length_bias(judgments, reps=reps, seed=seed)
    if any(j.model_a is not None and j.model_b is not None for j in judgments):
        sp = self_preference(judgments, reps=reps, seed=seed)
        if sp["n_used"]:
            out["self_preference"] = sp
    if any(j.human is not None for j in judgments):
        out["agreement"] = agreement(judgments, ties=ties, reps=reps, seed=seed)
    return out
