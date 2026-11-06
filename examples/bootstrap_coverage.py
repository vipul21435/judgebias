"""Check the coverage of the percentile bootstrap intervals by simulation.

For each setting, 200 datasets are drawn from a judge whose true value is
known, and the script prints how often the 95 % interval that ``judgebias``
reports contains it. Three statistics are checked:

- first-position rate: each prompt shown once, in a random order, and the
  first-shown answer picked with the true rate;
- Cohen's kappa: one human label per prompt (A, B or tie with probabilities
  0.45, 0.45, 0.10); the judge copies it with probability 0.6 and otherwise
  picks one of the three labels uniformly, which fixes the true kappa;
- length slope: log length ratios drawn from N(0, 0.5^2), and A wins with
  probability sigmoid(1.0 * log ratio), so the true slope is 1.0.

The first-position settings, with the same seeds, also run in the test suite.

    uv run python examples/bootstrap_coverage.py
"""

from __future__ import annotations

import numpy as np

from judgebias import Judgment
from judgebias.measures import _by_pair, _first_win_rate, _slope
from judgebias.stats import bootstrap, cohen_kappa

RUNS = 200
REPS = 400
HUMAN = (0.45, 0.45, 0.10)
COPY = 0.6
LABELS = ("A", "B", "tie")


def coverage(n_prompts: int, rate: float, runs: int = RUNS, reps: int = REPS) -> float:
    """Coverage of the first-position interval."""
    rng = np.random.default_rng(1)
    hits = 0
    for r in range(runs):
        js = []
        for i in range(n_prompts):
            first = rng.random() < rate
            order = "AB" if rng.random() < 0.5 else "BA"
            shown_first = "A" if order == "AB" else "B"
            other = "B" if shown_first == "A" else "A"
            js.append(Judgment(pair_id=f"p{i}", verdict=shown_first if first else other,
                               order=order))
        est = bootstrap(_by_pair(js), _first_win_rate, reps=reps, seed=r)
        hits += est.low <= rate <= est.high
    return hits / runs


def true_kappa() -> float:
    pi = np.asarray(HUMAN)
    po = COPY + (1 - COPY) / 3
    pe = float(np.sum(pi * (pi * COPY + (1 - COPY) / 3)))
    return (po - pe) / (1 - pe)


def _kappa(groups) -> float:
    rows = [j for g in groups for j in g]
    return cohen_kappa([j.verdict for j in rows], [j.human for j in rows])


def kappa_coverage(n_prompts: int, runs: int = RUNS, reps: int = REPS) -> float:
    rng = np.random.default_rng(2)
    truth = true_kappa()
    hits = 0
    for r in range(runs):
        js = []
        for i in range(n_prompts):
            human = LABELS[rng.choice(3, p=HUMAN)]
            judge = human if rng.random() < COPY else LABELS[rng.integers(3)]
            js.append(Judgment(pair_id=f"p{i}", verdict=judge, human=human))
        est = bootstrap(_by_pair(js), _kappa, reps=reps, seed=r)
        hits += est.low <= truth <= est.high
    return hits / runs


def slope_coverage(n_prompts: int, slope: float = 1.0, runs: int = RUNS,
                   reps: int = REPS) -> float:
    rng = np.random.default_rng(3)
    hits = 0
    for r in range(runs):
        js = []
        for i in range(n_prompts):
            x = rng.normal(0.0, 0.5)
            len_a, len_b = int(round(300 * np.exp(x / 2))), int(round(300 * np.exp(-x / 2)))
            ratio = np.log((len_a + 1) / (len_b + 1))
            a_wins = rng.random() < 1.0 / (1.0 + np.exp(-slope * ratio))
            js.append(Judgment(pair_id=f"p{i}", verdict="A" if a_wins else "B",
                               len_a=len_a, len_b=len_b))
        est = bootstrap(_by_pair(js), _slope, reps=reps, seed=r)
        hits += est.low <= slope <= est.high
    return hits / runs


def main() -> None:
    for n_prompts, rate in ((60, 0.6), (30, 0.85), (200, 0.65)):
        print(f"first-position {n_prompts:4d} prompts, true rate {rate:.2f}: "
              f"covered in {coverage(n_prompts, rate):.3f} of {RUNS} datasets")
    for n_prompts in (40, 150):
        print(f"kappa          {n_prompts:4d} prompts, true kappa {true_kappa():.3f}: "
              f"covered in {kappa_coverage(n_prompts):.3f} of {RUNS} datasets")
    for n_prompts in (40, 150):
        print(f"length slope   {n_prompts:4d} prompts, true slope 1.000: "
              f"covered in {slope_coverage(n_prompts):.3f} of {RUNS} datasets")


if __name__ == "__main__":
    main()
