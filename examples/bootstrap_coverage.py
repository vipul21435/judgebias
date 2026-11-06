"""Check the coverage of the percentile bootstrap interval by simulation.

For each setting, 200 datasets are drawn from a judge whose true
first-position rate is known; each prompt is shown once, in a random order.
The script prints how often the 95 % interval that ``judgebias`` reports
contains the true rate. The same check, with the same seeds, runs in the
test suite.

    uv run python examples/bootstrap_coverage.py
"""

from __future__ import annotations

import numpy as np

from judgebias import Judgment
from judgebias.measures import _by_pair, _first_win_rate
from judgebias.stats import bootstrap


def coverage(n_prompts: int, rate: float, runs: int = 200, reps: int = 400) -> float:
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


def main() -> None:
    for n_prompts, rate in ((60, 0.6), (30, 0.85), (200, 0.65)):
        print(f"{n_prompts:4d} prompts, true rate {rate:.2f}: "
              f"95 % interval covered it in {coverage(n_prompts, rate):.3f} of 200 datasets")


if __name__ == "__main__":
    main()
