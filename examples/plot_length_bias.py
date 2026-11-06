"""Plot verdict against length ratio for two judgment files, with the logistic fits.

Each file is binned by x = log((len_a + 1) / (len_b + 1)) into equal-count bins;
a point is the share of decided verdicts in the bin that went to A, with a 95 %
Wilson interval, and the curve is the logistic fit that ``judgebias audit``
reports. Needs matplotlib (not a dependency of judgebias):

    uv run --with matplotlib python examples/plot_length_bias.py \\
        GPT4.jsonl HUMANS.jsonl docs/length_bias.png --labels "GPT-4 judge" "Human experts"
"""

from __future__ import annotations

import argparse
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from judgebias import load_jsonl, logistic_fit  # noqa: E402
from judgebias.measures import distinct  # noqa: E402

COLOURS = ("#2a78d6", "#eb6834")
INK = "#0b0b0b"
MUTED = "#52514e"
SURFACE = "#fcfcfb"


def xy(path: str) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for j in distinct(load_jsonl(path)):
        if j.verdict == "tie" or j.len_a is None or j.len_b is None:
            continue
        xs.append(math.log((j.len_a + 1) / (j.len_b + 1)))
        ys.append(1.0 if j.verdict == "A" else 0.0)
    return np.asarray(xs), np.asarray(ys)


def wilson(k: float, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return mid - half, mid + half


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("first")
    ap.add_argument("second")
    ap.add_argument("out")
    ap.add_argument("--labels", nargs=2, default=("first", "second"))
    ap.add_argument("--bins", type=int, default=8)
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    grid = np.linspace(-2.5, 2.5, 200)
    for path, label, colour, shift in zip(
        (args.first, args.second), args.labels, COLOURS, (-0.03, 0.03), strict=True
    ):
        x, y = xy(path)
        a, b = logistic_fit(x, y)
        edges = np.quantile(x, np.linspace(0, 1, args.bins + 1))
        centres, rates, lows, highs = [], [], [], []
        for lo, hi in zip(edges[:-1], edges[1:], strict=True):
            m = (x >= lo) & (x <= hi)
            n = int(m.sum())
            if n == 0:
                continue
            k = float(y[m].sum())
            wl, wh = wilson(k, n)
            centres.append(float(np.median(x[m])) + shift)
            rates.append(k / n)
            lows.append(k / n - wl)
            highs.append(wh - k / n)
        ax.errorbar(centres, rates, yerr=[lows, highs], fmt="o", ms=5, color=colour,
                    elinewidth=1.2, capsize=0, zorder=3)
        ax.plot(grid, 1 / (1 + np.exp(-(a + b * grid))), color=colour, lw=2,
                label=f"{label}: slope {b:.2f}, n={len(x)}", zorder=2)
    ax.axhline(0.5, color=MUTED, lw=0.8, ls=":", zorder=1)
    ax.axvline(0.0, color=MUTED, lw=0.8, ls=":", zorder=1)
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(0, 1)
    ax.set_xlabel("log((len A + 1) / (len B + 1)), characters", color=INK)
    ax.set_ylabel("share of decided verdicts for A", color=INK)
    ax.set_title("MT-Bench pairs: verdict against answer length", color=INK, loc="left")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED)
    ax.grid(axis="y", color="#e6e5e1", lw=0.8, zorder=0)
    ax.legend(frameon=False, loc="upper left", labelcolor=INK)
    fig.tight_layout()
    fig.savefig(args.out, facecolor=SURFACE)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
