import math

import numpy as np
import pytest

from judgebias import Judgment, length_bias, logistic_fit, logistic_fit_multi


def _equal_lengths(n=40, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        for order in ("AB", "BA"):
            out.append(Judgment(pair_id=f"p{i}", verdict=str(rng.choice(["A", "B"])),
                                order=order, len_a=100, len_b=100,
                                human=str(rng.choice(["A", "B"]))))
    return out


def test_slope_is_nan_when_lengths_do_not_vary():
    rep = length_bias(_equal_lengths(), reps=50)
    assert math.isnan(rep["slope"]["value"])
    assert math.isnan(rep["slope"]["low"]) and math.isnan(rep["slope"]["high"])
    assert math.isnan(rep["slope_given_human"]["value"])


def test_logistic_fit_constant_x_gives_nan_slope():
    a, b = logistic_fit(np.zeros(10), np.array([0, 1] * 5))
    assert math.isnan(b)


def test_logistic_fit_multi_refuses_non_finite_x():
    with pytest.raises(ValueError, match="finite"):
        logistic_fit_multi(np.array([[0.0], [np.nan], [1.0]]), np.array([0.0, 1.0, 1.0]))


@pytest.mark.parametrize("field", ["len_a", "len_b"])
def test_negative_lengths_are_refused(field):
    with pytest.raises(ValueError, match=field):
        Judgment(pair_id="p", verdict="A", **{field: -2})
