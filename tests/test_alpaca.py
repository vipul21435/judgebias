"""AlpacaFarm and AlpacaEval conversion on hand-written rows shaped like the public files."""

import json

import pytest

from judgebias import alpaca
from judgebias.alpaca import (
    convert_cross_judge,
    convert_cross_pairs,
    convert_eval,
    convert_eval_cross,
    convert_farm,
    preference,
)
from judgebias.cli import main
from judgebias.data import judgment_from_dict, load_jsonl
from judgebias.measures import agreement, position_bias


def farm(instr, o1, o2, pref, inp=""):
    return {"instruction": instr, "input": inp, "output_1": o1, "output_2": o2, "preference": pref}


def cross(instr, o1, o2, pref, who):
    return {
        "instruction": instr,
        "output_1": o1,
        "output_2": o2,
        "preference": pref,
        "annotator_index": who,
        "dataset": "toy",
        "datasplit": "eval",
    }


def ev(instr, gen2, o2, pref, judge):
    return {
        "instruction": instr,
        "output_1": "baseline answer",
        "generator_1": "text_davinci_003",
        "output_2": o2,
        "generator_2": gen2,
        "annotator": judge,
        "preference": pref,
    }


@pytest.mark.parametrize(
    "value,expected",
    [("1", "A"), ("2", "B"), (1.0, "A"), ("2.0", "B"), ("1.5", "tie"), (1.5, "tie"),
     ("0", None), (0.0, None), (None, None), ("None", None), (float("nan"), None)],
)
def test_preference_values(value, expected):
    assert preference(value) == expected


@pytest.mark.parametrize("value", ["3", "-1", "A", "yes"])
def test_preference_refuses_other_values(value):
    with pytest.raises(ValueError):
        preference(value)


def test_convert_farm_joins_on_text_and_keeps_human_names():
    humans = [
        farm("say hi", "hi", "hello there", "2"),
        farm("say hi", "hi", "hello there", "1", inp="loudly"),  # different input: other pair
        farm("count", "1 2 3", "one two", "1"),
        farm("unjudged", "x", "y", "1"),
    ]
    judge = [
        farm("say hi", "hi", "hello there", "2"),
        farm("say hi", "hi", "hello there", "2", inp="loudly"),
        farm("count", "one two", "1 2 3", "2"),  # stored the other way round
    ]
    out = convert_farm(judge, humans)
    assert [r["pair_id"] for r in out] == ["af00001", "af00002", "af00003"]
    assert [r["verdict"] for r in out] == ["B", "B", "A"]
    assert [r["order"] for r in out] == ["AB", "AB", "BA"]
    assert [r["human"] for r in out] == ["B", "A", "A"]
    assert out[0]["len_a"] == 2 and out[0]["len_b"] == 11
    for r in out:
        judgment_from_dict(r)


def test_convert_farm_pair_found_both_ways_is_a_flip_when_slot_wins_twice():
    humans = [farm("q", "short", "longer one", "2")]
    judge = [farm("q", "short", "longer one", "1"), farm("q", "longer one", "short", "1")]
    out = convert_farm(judge, humans)
    assert [(r["order"], r["verdict"]) for r in out] == [("AB", "A"), ("BA", "B")]
    pos = position_bias([judgment_from_dict(r) for r in out])
    assert pos["flip_rate"]["value"] == 1.0
    assert pos["first_both_rate"]["value"] == 1.0


def test_convert_farm_skips_unreadable_preferences():
    humans = [farm("q", "a", "b", None), farm("r", "a", "b", "1")]
    judge = [farm("q", "a", "b", "1"), farm("r", "a", "b", "0")]
    assert convert_farm(judge, humans) == []


def test_cross_pairs_one_row_per_two_annotators():
    rows = [
        cross("q", "a", "bb", "1", 3),
        cross("q", "a", "bb", "2", 5),
        cross("q", "a", "bb", "1", 7),
        cross("r", "c", "d", "2", 3),  # one annotator only: no pair
        cross("s", "e", "f", "1", 4),
        cross("s", "e", "f", "2", 4),  # same annotator twice: not a pair
    ]
    out = convert_cross_pairs(rows)
    assert len(out) == 3
    assert {(r["judge"], r["human_judge"]) for r in out} == {
        ("annotator_3", "annotator_5"),
        ("annotator_3", "annotator_7"),
        ("annotator_5", "annotator_7"),
    }
    assert {r["pair_id"] for r in out} == {"ax0001"}
    agr = agreement([judgment_from_dict(r) for r in out])
    assert agr["raw"]["value"] == pytest.approx(1 / 3)


def test_cross_judge_matches_without_input_field():
    judge = [farm("q", "a", "bb", "1", inp="ignored"), farm("z", "a", "b", "2")]
    rows = [
        cross("q", "a", "bb", "1", 3),
        cross("q", "a", "bb", "2", 5),
        cross("w", "a", "b", "1", 3),
    ]
    out = convert_cross_judge(judge, rows)
    assert [(r["verdict"], r["human"], r["human_judge"]) for r in out] == [
        ("A", "A", "annotator_3"),
        ("A", "B", "annotator_5"),
    ]


def test_convert_eval_names_models_and_counts_skipped():
    g = [
        ev("q1", "gpt4", "long answer here", "2.0", "alpaca_eval_gpt4"),
        ev("q2", "claude", "baseline answer", "0.0", "alpaca_eval_gpt4"),  # identical outputs
        ev("q3", "claude", "other", None, "alpaca_eval_gpt4"),
    ]
    c = [ev("q1", "gpt4", "long answer here", "1", "claude")]
    rows, skipped = convert_eval(g, c)
    assert skipped == 2
    assert [(r["judge"], r["verdict"], r["pair_id"]) for r in rows] == [
        ("alpaca_eval_gpt4", "B", "ae00001"),
        ("claude", "A", "ae00001"),
    ]
    assert rows[0]["model_a"] == "text_davinci_003" and rows[0]["model_b"] == "gpt4"
    assert rows[0]["len_b"] == len("long answer here")


def test_convert_eval_cross_puts_other_judge_in_human_field():
    c = [ev("q1", "claude", "x", "2", "claude"), ev("q2", "claude", "y", "1", "claude"),
         ev("q3", "gpt4", "z", "1", "claude")]
    g = [ev("q1", "claude", "x", "1.0", "alpaca_eval_gpt4"),
         ev("q2", "claude", "y", "1.0", "alpaca_eval_gpt4")]
    out = convert_eval_cross(c, g)
    assert [(r["verdict"], r["human"], r["human_judge"]) for r in out] == [
        ("B", "A", "alpaca_eval_gpt4"),
        ("A", "A", "alpaca_eval_gpt4"),
    ]


def test_cli_alpaca_reads_local_files(tmp_path, monkeypatch, capsys):
    data = tmp_path / "data"
    data.mkdir()
    files = {
        "farm_human": [farm("q", "a", "bb", "2"), farm("r", "ccc", "d", "1")],
        "farm_gpt4": [farm("q", "a", "bb", "2"), farm("r", "ccc", "d", "2")],
        "farm_cross": [cross("q", "a", "bb", "2", 1), cross("q", "a", "bb", "1", 2)],
        "eval_gpt4": [ev("q", "claude", "x", "2", "alpaca_eval_gpt4")],
        "eval_claude": [ev("q", "claude", "x", "2", "claude")],
    }
    for name, rows in files.items():
        (data / f"{name}.json").write_text(json.dumps(rows), encoding="utf-8")

    def no_network(*args, **kwargs):
        raise AssertionError("tried to download")

    monkeypatch.setattr(alpaca.urllib.request, "urlopen", no_network)
    outs = {k: tmp_path / f"{k}.jsonl" for k in ("fj", "cj", "hp", "ej", "cg", "gc")}
    code = main([
        "alpaca", "--data-dir", str(data),
        "--farm-judge", str(outs["fj"]), "--cross-judge", str(outs["cj"]),
        "--human-pairs", str(outs["hp"]), "--eval-judges", str(outs["ej"]),
        "--eval-claude-vs-gpt4", str(outs["cg"]), "--eval-gpt4-vs-claude", str(outs["gc"]),
    ])
    assert code == 0
    assert len(load_jsonl(outs["fj"])) == 2
    assert len(load_jsonl(outs["cj"])) == 2
    assert len(load_jsonl(outs["hp"])) == 1
    assert len(load_jsonl(outs["ej"])) == 2
    assert load_jsonl(outs["cg"])[0].judge == "claude"
    assert load_jsonl(outs["gc"])[0].judge == "alpaca_eval_gpt4"
    out = capsys.readouterr().out
    assert "rows read: farm_human 2, farm_gpt4 2, farm_cross 2, eval_gpt4 1, eval_claude 1" in out
    assert "wrote 2 GPT-4 judgments with human labels" in out


def test_cli_alpaca_needs_an_output(tmp_path, capsys):
    assert main(["alpaca", "--data-dir", str(tmp_path)]) == 2
    assert "at least one output" in capsys.readouterr().err


def test_cli_alpaca_reports_bad_file(tmp_path, capsys):
    (tmp_path / "farm_cross.json").write_text('{"not": "a list"}', encoding="utf-8")
    assert main(["alpaca", "--data-dir", str(tmp_path), "--human-pairs", str(tmp_path / "o")]) == 2
    assert "expected a JSON list" in capsys.readouterr().err


def test_download_skips_present_files(tmp_path, monkeypatch):
    (tmp_path / "farm_cross.json").write_text("[]", encoding="utf-8")
    calls = []

    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return b"[]"

    def fake(url, timeout):
        calls.append(url)
        return Resp()

    monkeypatch.setattr(alpaca.urllib.request, "urlopen", fake)
    paths = alpaca.download(tmp_path, ["farm_cross", "eval_claude"])
    assert calls == [alpaca.URLS["eval_claude"]]
    assert paths["eval_claude"].read_text() == "[]"
    assert not list(tmp_path.glob("*.part"))
