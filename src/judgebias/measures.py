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
from .stats import Estimate, bootstrap, cohen_kappa, logistic_fit, logistic_fit_multi


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


def conflicting_copies(judgments: Sequence[Judgment]) -> dict[str, int]:
    """Count (pair, order, judge, repeat) keys whose rows differ in verdict.

    :func:`distinct` keeps the first row of each key, which is right for
    copies of one verdict. Rows that share a key but not a verdict are more
    likely separate samples written without a ``repeat`` field; only the
    first of them reaches the position and length measures. ``n_keys`` is the
    number of such keys and ``n_rows`` the rows beyond the first under them.
    """
    rows: dict[tuple[str, str, str, int], list[str]] = defaultdict(list)
    for j in judgments:
        rows[(j.pair_id, j.order, j.judge, j.repeat)].append(j.verdict)
    bad = [v for v in rows.values() if len(set(v)) > 1]
    return {"n_keys": len(bad), "n_rows": sum(len(v) - 1 for v in bad)}


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


def _combined_flip_rate(groups: Sequence[Sequence[Judgment]]) -> float:
    """Share of combined-order verdicts flagged as flipped."""
    flags = [j.flipped for j in _flat(groups) if j.flipped is not None]
    return _nan_if_empty(sum(flags), len(flags))


def _swap_units(g: Sequence[Judgment]) -> list[tuple[Judgment, Judgment]]:
    """The (AB, BA) verdict pairs of one prompt, matched by judge and repeat.

    A flip compares one judge with itself: the AB sample of a given judge and
    repeat is set against the BA sample of the same judge and repeat. Rows by
    different judges, or different repeats, are never compared, so a file
    with several judges that disagree with each other shows no false flips.
    """
    ab: dict[tuple[str, int], Judgment] = {}
    ba: dict[tuple[str, int], Judgment] = {}
    for j in g:
        if j.flipped is not None:
            continue
        side = ab if j.order == "AB" else ba
        side.setdefault((j.judge, j.repeat), j)
    return [(ab[k], ba[k]) for k in ab if k in ba]


def _flip_rate(groups: Sequence[Sequence[Judgment]]) -> float:
    """Share of (pair, judge, repeat) units seen in both orders whose verdict changed."""
    units = [u for g in groups for u in _swap_units(g)]
    flips = sum(a.verdict != b.verdict for a, b in units)
    return _nan_if_empty(flips, len(units))


def _consistent_first_rate(groups: Sequence[Sequence[Judgment]]) -> float:
    """Among flipped units, the share where the judge took the first answer both times."""
    flipped = [
        (a, b) for g in groups for a, b in _swap_units(g) if a.verdict != b.verdict
    ]
    first_both = sum(bool(a.picked_first and b.picked_first) for a, b in flipped)
    return _nan_if_empty(first_both, len(flipped))


def position_bias(
    judgments: Sequence[Judgment], *, reps: int = 1000, seed: int = 0
) -> dict[str, Any]:
    """First-position win rate and flip rate under swapped presentation.

    ``first_win_rate`` is the share of decided verdicts that picked whichever
    answer was shown first; 0.5 is unbiased. ``flip_rate`` needs pairs judged
    in both orders by the same judge: over (pair, judge, repeat) units with an
    AB and a BA sample, the share whose verdict changed when the order
    changed. ``n_pairs_both_orders`` counts those units (one per pair when
    there is one judge and no repeats). Verdicts by different judges are never
    compared with each other. ``first_both_rate`` is, among flipped units, the
    share where the judge
    chose the first-shown answer both times (a pure position preference) as
    opposed to the second both times. Rows whose verdict already combines both
    orders (``flipped`` set) are left out of those three and give
    ``combined_flip_rate`` instead: the share flagged as flipped. Copied rows
    count once (see :func:`distinct`).
    """
    n_rows = len(judgments)
    judgments = distinct(judgments)
    groups = _by_pair(judgments)
    single = [j for j in judgments if j.flipped is None]
    picks = [j.picked_first for j in single]
    both = sum(len(_swap_units(g)) for g in groups)
    n_combined = len(judgments) - len(single)
    out: dict[str, Any] = {
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
        "n_combined": n_combined,
    }
    if n_combined:
        out["combined_flip_rate"] = bootstrap(
            groups, _combined_flip_rate, reps=reps, seed=seed
        ).as_dict()
    return out


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


def _adjusted_rows(judgments: Sequence[Judgment]) -> tuple[np.ndarray, np.ndarray]:
    """Covariates (log length ratio, human label as +1/-1) and judge verdicts."""
    xs: list[tuple[float, float]] = []
    ys: list[float] = []
    for j in judgments:
        if j.verdict == "tie" or j.human in (None, "tie") or j.len_a is None or j.len_b is None:
            continue
        xs.append((float(np.log((j.len_a + 1) / (j.len_b + 1))), 1.0 if j.human == "A" else -1.0))
        ys.append(1.0 if j.verdict == "A" else 0.0)
    return np.asarray(xs, dtype=float).reshape(-1, 2), np.asarray(ys)


def _adjusted_slope(groups: Sequence[Sequence[Judgment]]) -> float:
    x, y = _adjusted_rows(_flat(groups))
    if x.shape[0] < 3 or np.all(y == y[0]) or np.ptp(x[:, 0]) == 0:
        return float("nan")
    return float(logistic_fit_multi(x, y)[1])


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
    When every pair has the same length ratio the slope is not identified and
    is nan (shown as n/a), not 0.
    Both come with a bootstrap interval over prompts. Copied rows count once
    (see :func:`distinct`).

    When human labels are present, ``slope_given_human`` is the length
    coefficient in a fit that also has the human label (+1 for A, -1 for B)
    as a covariate, on rows where neither side said tie. It asks whether
    length moves the judge beyond what it moves the humans: answers that are
    longer because they are better push the human label too, and that part
    goes to the human coefficient. Every labelled row is used, so a verdict
    with several human labels enters once per label.
    """
    all_rows = judgments
    judgments = distinct(judgments)
    groups = _by_pair(judgments)
    x, y = _length_rows(judgments)
    intercept, slope = logistic_fit(x, y) if x.size >= 2 else (float("nan"), float("nan"))
    out: dict[str, Any] = {
        "n_used": int(x.size),
        "longer_win_rate": bootstrap(groups, _longer_win_rate, reps=reps, seed=seed).as_dict(),
        "intercept": intercept,
        "slope": Estimate(
            slope,
            *_interval(groups, _slope, reps, seed),
            n=len(groups),
        ).as_dict(),
    }
    xa, _ = _adjusted_rows(all_rows)
    if xa.shape[0]:
        labelled = _by_pair(all_rows)
        out["n_adjusted"] = int(xa.shape[0])
        out["slope_given_human"] = bootstrap(
            labelled, _adjusted_slope, reps=reps, seed=seed
        ).as_dict()
    return out


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
    judge's family (see :func:`family`; ``judge="gpt-4"`` and
    ``model_a="gpt-3.5"`` share family ``gpt``).
    Each row is matched against its own judge's family; ``judge_family``
    lists every judge family in the input, comma-separated.
    ``own_win_rate`` above the judge's overall win rate for that family on
    human labels suggests self-preference; the human comparison is only
    reported when human labels exist, together with ``excess``: the judge's
    rate minus the human rate, with a bootstrap interval over prompts in
    which both rates are recomputed on the same resample. Each rate leaves
    out its own ties.
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
        "judge_family": ", ".join(sorted({str(family(j.judge)) for j in judgments})) or None,
        "n_used": n_used,
        "own_win_rate": bootstrap(
            groups, lambda g: own(g, False), reps=reps, seed=seed
        ).as_dict(),
    }
    if any(j.human is not None for j in judgments):
        out["own_win_rate_human"] = bootstrap(
            groups, lambda g: own(g, True), reps=reps, seed=seed
        ).as_dict()
        out["excess"] = bootstrap(
            groups, lambda g: own(g, False) - own(g, True), reps=reps, seed=seed
        ).as_dict()
    return out


def family(model: str | None) -> str | None:
    """Lower-cased model name up to the first ``/``, ``-``, ``_`` or ``:``."""
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
    ``n_conflicting_keys`` counts keys whose rows differ in verdict (see
    :func:`conflicting_copies`); such rows are probably repeats that lack a
    ``repeat`` field, and only the first of them is used here.
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
        "n_conflicting_keys": conflicting_copies(judgments)["n_keys"],
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
        "conflicting_copies": conflicting_copies(judgments),
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
