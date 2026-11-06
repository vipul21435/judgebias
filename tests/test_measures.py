import json
import math

import numpy as np
import pytest

from judgebias import (
    Judgment,
    agreement,
    audit,
    cohen_kappa,
    judgment_from_dict,
    length_bias,
    load_jsonl,
    logistic_fit,
    position_bias,
    repeat_consistency,
    self_preference,
)
from judgebias.cli import main
from judgebias.synth import simulate, to_rows


def j(pair, verdict, order="AB", **kw):
    return Judgment(pair_id=pair, verdict=verdict, order=order, **kw)


# --- data ------------------------------------------------------------------


def test_judgment_validation():
    with pytest.raises(ValueError):
        Judgment(pair_id="x", verdict="C")
    with pytest.raises(ValueError):
        Judgment(pair_id="x", verdict="A", order="XY")
    with pytest.raises(ValueError):
        Judgment(pair_id="x", verdict="A", human="maybe")


def test_picked_first_follows_order():
    assert j("p", "A", "AB").picked_first is True
    assert j("p", "A", "BA").picked_first is False
    assert j("p", "tie").picked_first is None


def test_judgment_from_dict_aliases_and_lengths():
    row = {
        "question_id": 7,
        "winner": "model_b",
        "answer_a": "short",
        "answer_b": "a much longer answer",
        "gold": "tie",
        "note": "kept",
    }
    jd = judgment_from_dict(row)
    assert jd.pair_id == "7"
    assert jd.verdict == "B"
    assert jd.human == "tie"
    assert (jd.len_a, jd.len_b) == (5, 20)
    assert jd.extra == {"note": "kept"}
    with pytest.raises(ValueError):
        judgment_from_dict({"verdict": "A"})
    with pytest.raises(ValueError):
        judgment_from_dict({"pair_id": "1", "verdict": "strongly A"})


def test_load_jsonl_reports_bad_line(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"pair_id": "1", "verdict": "A"}\n\n{not json\n')
    with pytest.raises(ValueError, match="bad JSON"):
        load_jsonl(path)


# --- position --------------------------------------------------------------


def test_position_bias_unbiased_judge():
    # same verdict in both orders: no flips, first win rate 0.5
    js = []
    for i in range(40):
        v = "A" if i % 2 else "B"
        js += [j(f"p{i}", v, "AB"), j(f"p{i}", v, "BA")]
    p = position_bias(js, reps=50)
    assert p["first_win_rate"]["value"] == pytest.approx(0.5)
    assert p["flip_rate"]["value"] == 0.0
    assert p["n_pairs_both_orders"] == 40
    assert math.isnan(p["first_both_rate"]["value"])


def test_position_bias_always_first():
    js = []
    for i in range(30):
        js += [j(f"p{i}", "A", "AB"), j(f"p{i}", "B", "BA")]
    p = position_bias(js, reps=50)
    assert p["first_win_rate"]["value"] == 1.0
    assert p["flip_rate"]["value"] == 1.0
    assert p["first_both_rate"]["value"] == 1.0
    assert p["first_win_rate"]["low"] == 1.0 == p["first_win_rate"]["high"]


def test_position_bias_without_swaps():
    p = position_bias([j("p1", "A"), j("p2", "B"), j("p3", "tie")], reps=10)
    assert p["n_pairs_both_orders"] == 0
    assert math.isnan(p["flip_rate"]["value"])
    assert p["n_ties"] == 1


# --- length ----------------------------------------------------------------


def test_length_bias_recovers_sign():
    rng = np.random.default_rng(1)
    js = []
    for i in range(300):
        la, lb = int(rng.integers(50, 2000)), int(rng.integers(50, 2000))
        v = "A" if la > lb else "B"
        js.append(j(f"p{i}", v, len_a=la, len_b=lb))
    ln = length_bias(js, reps=50)
    assert ln["longer_win_rate"]["value"] == 1.0
    assert ln["slope"]["value"] > 2.0
    assert ln["slope"]["low"] > 0.0


def test_length_bias_no_signal_slope_near_zero():
    rng = np.random.default_rng(2)
    js = []
    for i in range(400):
        la, lb = int(rng.integers(50, 2000)), int(rng.integers(50, 2000))
        js.append(j(f"p{i}", "A" if rng.random() < 0.5 else "B", len_a=la, len_b=lb))
    ln = length_bias(js, reps=100)
    assert ln["slope"]["low"] < 0.0 < ln["slope"]["high"]
    assert 0.4 < ln["longer_win_rate"]["value"] < 0.6


def test_length_bias_skips_missing_and_ties():
    ln = length_bias([j("p1", "A"), j("p2", "tie", len_a=1, len_b=9)], reps=5)
    assert ln["n_used"] == 0
    assert math.isnan(ln["slope"]["value"])


# --- stats -----------------------------------------------------------------


def test_cohen_kappa_known_values():
    assert cohen_kappa(["A", "B", "A", "B"], ["A", "B", "A", "B"]) == 1.0
    assert cohen_kappa(["A", "A", "B", "B"], ["A", "B", "A", "B"]) == pytest.approx(0.0)
    assert cohen_kappa(["A", "A"], ["A", "A"]) == 1.0
    assert math.isnan(cohen_kappa([], []))
    with pytest.raises(ValueError):
        cohen_kappa(["A"], [])
    # 2x2 by hand: po = 35/50 = 0.7, pe = (25*30 + 25*20)/2500 = 0.5, kappa = 0.2/0.5
    a = ["y"] * 25 + ["n"] * 25
    b = ["y"] * 20 + ["n"] * 5 + ["y"] * 10 + ["n"] * 15
    assert cohen_kappa(a, b) == pytest.approx(0.4)


def test_logistic_fit_recovers_parameters():
    rng = np.random.default_rng(3)
    x = rng.normal(size=5000)
    p = 1 / (1 + np.exp(-(0.5 + 1.5 * x)))
    y = (rng.random(5000) < p).astype(float)
    a, b = logistic_fit(x, y)
    assert a == pytest.approx(0.5, abs=0.1)
    assert b == pytest.approx(1.5, abs=0.15)


def test_logistic_fit_degenerate():
    assert logistic_fit(np.array([]), np.array([])) == (pytest.approx(math.nan, nan_ok=True),) * 2
    a, b = logistic_fit(np.array([1.0, 2.0]), np.array([1.0, 1.0]))
    assert a == math.inf and b == 0.0


# --- self-preference -------------------------------------------------------


def test_self_preference_counts_only_mixed_pairs():
    js = [
        j("p1", "A", judge="gamma-70b", model_a="gamma-7b", model_b="beta-13b", human="B"),
        j("p2", "B", judge="gamma-70b", model_a="alpha-7b", model_b="gamma-70b", human="B"),
        j("p3", "A", judge="gamma-70b", model_a="alpha-7b", model_b="beta-13b", human="A"),
        j("p4", "A", judge="gamma-70b", model_a="gamma-7b", model_b="gamma-70b", human="A"),
    ]
    s = self_preference(js, reps=20)
    assert s["judge_family"] == "gamma"
    assert s["n_used"] == 2
    assert s["own_win_rate"]["value"] == 1.0
    assert s["own_win_rate_human"]["value"] == 0.5
    assert s["excess"] == pytest.approx(0.5)


# --- agreement -------------------------------------------------------------


def test_agreement_ties_keep_vs_drop():
    js = [
        j("p1", "A", human="A"),
        j("p2", "B", human="B"),
        j("p3", "tie", human="A"),
        j("p4", "A", human="tie"),
        j("p5", "B", human="A"),
        j("p6", "A"),
    ]
    keep = agreement(js, ties="keep", reps=20)
    drop = agreement(js, ties="drop", reps=20)
    assert keep["n_labelled"] == 5 and drop["n_labelled"] == 3
    assert keep["raw"]["value"] == pytest.approx(2 / 5)
    assert drop["raw"]["value"] == pytest.approx(2 / 3)
    assert keep["n_human_ties"] == 1 and keep["n_judge_ties"] == 1
    assert keep["confusion"]["judge=tie,human=A"] == 1
    with pytest.raises(ValueError):
        agreement(js, ties="ignore")


def test_agreement_bootstrap_interval_contains_point():
    js = simulate(80, seed=4)
    a = agreement(js, reps=200)
    assert a["kappa"]["low"] <= a["kappa"]["value"] <= a["kappa"]["high"]
    assert a["raw"]["low"] <= a["raw"]["value"] <= a["raw"]["high"]


# --- repeat ----------------------------------------------------------------


def test_repeat_consistency():
    js = [
        j("p1", "A", repeat=0),
        j("p1", "A", repeat=1),
        j("p1", "B", repeat=2),
        j("p2", "B", repeat=0),
        j("p2", "B", repeat=1),
        j("p3", "A"),
    ]
    r = repeat_consistency(js)
    assert r["n_groups"] == 2
    assert r["unanimous_rate"] == 0.5
    assert r["pairwise_agreement"] == pytest.approx(2 / 4)
    assert math.isnan(repeat_consistency([j("p", "A")])["unanimous_rate"])


def test_repeat_consistency_ignores_copies_and_other_judges():
    # one verdict copied once per human label is a single sample, not a repeat
    copies = [j("p1", "A", human=h) for h in ("A", "B", "tie")]
    assert repeat_consistency(copies)["n_groups"] == 0
    # two different judges on the same pair are not repeats of each other
    judges = [j("p1", "A", judge="x"), j("p1", "B", judge="y")]
    assert repeat_consistency(judges)["n_groups"] == 0
    mixed = judges + [j("p1", "B", judge="x", repeat=1)]
    r = repeat_consistency(mixed)
    assert r["n_groups"] == 1 and r["unanimous_rate"] == 0.0


# --- simulation: known biases are detected ---------------------------------


def test_simulated_position_bias_is_detected():
    fair = position_bias(simulate(300, position=0.0, seed=5), reps=100)
    biased = position_bias(simulate(300, position=1.0, seed=5), reps=100)
    # a fair judge seen in both orders scores exactly 0.5 unless a verdict flips
    assert fair["first_win_rate"]["low"] <= 0.5 <= fair["first_win_rate"]["high"]
    assert biased["first_win_rate"]["low"] > 0.6
    assert biased["flip_rate"]["value"] > fair["flip_rate"]["value"]
    assert biased["first_both_rate"]["value"] > 0.9


def test_simulated_length_bias_is_detected():
    fair = length_bias(simulate(300, length=0.0, seed=6), reps=100)
    biased = length_bias(simulate(300, length=2.0, seed=6), reps=100)
    assert fair["slope"]["low"] < 0.0 < fair["slope"]["high"]
    assert biased["slope"]["low"] > 0.0
    assert biased["longer_win_rate"]["value"] > fair["longer_win_rate"]["value"]


def test_simulated_self_preference_is_detected():
    fair = self_preference(simulate(400, self_pref=0.0, seed=7), reps=100)
    biased = self_preference(simulate(400, self_pref=2.0, seed=7), reps=100)
    assert abs(fair["excess"]) < 0.15
    assert biased["excess"] > 0.3


def test_audit_omits_sections_without_data():
    rep = audit([j("p1", "A"), j("p2", "B")], reps=5)
    assert set(rep) == {"n_judgments", "judges", "position", "repeat"}
    full = audit(simulate(20, repeats=2, seed=8), reps=5)
    assert {"length", "self_preference", "agreement"} <= set(full)
    assert full["repeat"]["n_groups"] == 40


# --- cli -------------------------------------------------------------------


def test_cli_simulate_then_audit(tmp_path, capsys):
    path = tmp_path / "sim.jsonl"
    assert main(["simulate", str(path), "--pairs", "30", "--position", "0.5"]) == 0
    assert main(["audit", str(path), "--reps", "20"]) == 0
    out = capsys.readouterr().out
    assert "first-position win rate" in out
    assert "Cohen's kappa" in out
    assert main(["audit", str(path), "--reps", "20", "--json", "--ties", "drop"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["agreement"]["ties"] == "drop"
    assert report["n_judgments"] == 60


def test_cli_errors(tmp_path, capsys):
    assert main(["audit", str(tmp_path / "missing.jsonl")]) == 2
    empty = tmp_path / "empty.jsonl"
    empty.write_text("\n")
    assert main(["audit", str(empty)]) == 2
    assert "no judgments" in capsys.readouterr().err


def test_to_rows_round_trip(tmp_path):
    from judgebias import write_jsonl

    js = simulate(5, seed=9)
    path = tmp_path / "r.jsonl"
    write_jsonl(path, to_rows(js))
    assert load_jsonl(path) == js


def test_family_prefix_rule():
    from judgebias.measures import family

    assert family("gamma-70b") == "gamma"
    assert family("org/alpha_7b:chat") == "org"
    assert family(None) is None


def test_render_includes_repeat_section(capsys, tmp_path):
    path = tmp_path / "rep.jsonl"
    assert main(["simulate", str(path), "--pairs", "10", "--repeats", "3", "--seed", "2"]) == 0
    assert main(["audit", str(path), "--reps", "10"]) == 0
    out = capsys.readouterr().out
    assert "repeat consistency" in out
    assert "unanimous groups" in out


def test_position_and_length_count_copied_rows_once():
    from judgebias.measures import distinct

    one = [j("p1", "A", len_a=50, len_b=10), j("p2", "B", len_a=50, len_b=10)]
    copied = one + [j("p1", "A", len_a=50, len_b=10, human=h) for h in ("A", "B")]
    assert len(distinct(copied)) == 2
    pos = position_bias(copied, reps=10)
    assert pos["n_rows"] == 4 and pos["n_judgments"] == 2
    assert pos["first_win_rate"]["value"] == 0.5
    assert length_bias(copied, reps=10)["n_used"] == 2
    assert length_bias(copied, reps=10)["longer_win_rate"]["value"] == 0.5
