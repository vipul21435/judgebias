"""Small statistics helpers: bootstrap intervals, Cohen's kappa, logistic fit."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


@dataclass(frozen=True)
class Estimate:
    """A point estimate with a percentile bootstrap interval."""

    value: float
    low: float
    high: float
    n: int
    level: float = 0.95

    def as_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "low": self.low,
            "high": self.high,
            "n": self.n,
            "level": self.level,
        }


def bootstrap(
    items: Sequence[Any],
    statistic: Callable[[Sequence[Any]], float],
    *,
    reps: int = 1000,
    level: float = 0.95,
    seed: int = 0,
) -> Estimate:
    """Percentile bootstrap of ``statistic`` over resamples of ``items``.

    ``items`` are resampled as units, so pass one item per independent unit
    (for example one item per prompt, holding both orders of a pair together).
    With fewer than two items the interval collapses onto the point estimate.
    """
    n = len(items)
    if n == 0:
        return Estimate(float("nan"), float("nan"), float("nan"), 0, level)
    value = float(statistic(items))
    if n < 2 or reps <= 0:
        return Estimate(value, value, value, n, level)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(reps, n))
    draws = np.empty(reps)
    arr = np.asarray(items, dtype=object) if not isinstance(items, np.ndarray) else items
    for i in range(reps):
        draws[i] = statistic([arr[j] for j in idx[i]])
    draws = draws[~np.isnan(draws)]
    if draws.size == 0:
        return Estimate(value, value, value, n, level)
    alpha = (1.0 - level) / 2.0
    low, high = np.quantile(draws, [alpha, 1.0 - alpha])
    return Estimate(value, float(low), float(high), n, level)


def cohen_kappa(a: Sequence[str], b: Sequence[str]) -> float:
    """Cohen's kappa between two label sequences of equal length.

    Labels are compared as given; a tie label is one more category, so a
    judge that says tie where the human says A is a disagreement.
    Returns nan on empty input and 1.0 when both raters are constant and equal.
    """
    if len(a) != len(b):
        raise ValueError("label sequences differ in length")
    n = len(a)
    if n == 0:
        return float("nan")
    cats = sorted(set(a) | set(b))
    index = {c: i for i, c in enumerate(cats)}
    table = np.zeros((len(cats), len(cats)))
    for x, y in zip(a, b, strict=True):
        table[index[x], index[y]] += 1
    po = np.trace(table) / n
    pe = float(np.sum(table.sum(axis=1) * table.sum(axis=0)) / (n * n))
    if pe >= 1.0:
        return 1.0 if po >= 1.0 else 0.0
    return float((po - pe) / (1.0 - pe))


def logistic_fit(x: np.ndarray, y: np.ndarray, *, l2: float = 1e-4) -> tuple[float, float]:
    """Fit P(y=1) = sigmoid(a + b*x) by penalised maximum likelihood.

    A small L2 penalty keeps the fit finite when the classes are separable.
    Returns ``(intercept, slope)``. When every ``y`` is the same, or every
    ``x`` is, the slope is not identified and comes back as nan (with every
    ``y`` the same, the intercept is +inf or -inf), so that a bootstrap drops
    such a resample instead of counting it as a slope of zero.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size == 0:
        return float("nan"), float("nan")
    if np.all(y == y[0]):
        return (float("inf") if y[0] == 1 else float("-inf")), float("nan")
    if np.ptp(x) == 0:
        return float("nan"), float("nan")
    w = logistic_fit_multi(x[:, None], y, l2=l2)
    return float(w[0]), float(w[1])


def logistic_fit_multi(x: np.ndarray, y: np.ndarray, *, l2: float = 1e-4) -> np.ndarray:
    """Fit P(y=1) = sigmoid(w0 + x @ w[1:]) by penalised maximum likelihood.

    ``x`` has one row per observation and one column per covariate. The L2
    penalty applies to the covariate weights, not the intercept. Returns the
    weights with the intercept first, and warns (``RuntimeWarning``) if the
    optimiser reports that it did not converge.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 2 or x.shape[0] != y.shape[0]:
        raise ValueError("x must be 2-D with one row per element of y")
    if not np.all(np.isfinite(x)):
        raise ValueError("x must be finite")
    design = np.hstack([np.ones((x.shape[0], 1)), x])
    penalty = np.full(design.shape[1], l2)
    penalty[0] = 0.0

    def nll(w: np.ndarray) -> float:
        z = design @ w
        # log-likelihood of logistic regression via log-sum-exp
        ll = y * z - np.logaddexp(0.0, z)
        return float(-ll.sum() + 0.5 * np.sum(penalty * w * w))

    def grad(w: np.ndarray) -> np.ndarray:
        r = expit(design @ w) - y
        return design.T @ r + penalty * w

    res = minimize(nll, np.zeros(design.shape[1]), jac=grad, method="L-BFGS-B")
    if not res.success:
        warnings.warn(f"logistic fit did not converge: {res.message}", RuntimeWarning,
                      stacklevel=2)
    return np.asarray(res.x, dtype=float)
