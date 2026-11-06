"""Run mode: fake transport, verdict parsing, slot-to-name mapping, CLI."""

import json

import pytest

from judgebias.cli import main
from judgebias.data import load_jsonl, write_jsonl
from judgebias.judge import (
    HttpJudge,
    Pair,
    load_pairs,
    pair_from_dict,
    parse_verdict,
    run_pairs,
    slot_to_name,
)
from judgebias.measures import position_bias


def reply(text):
    return {"choices": [{"message": {"content": text}}]}


class FirstSlotTransport:
    """Pretends to be a server whose judge always prefers the first response."""

    def __init__(self):
        self.requests = []

    def __call__(self, url, headers, body, timeout):
        self.requests.append((url, headers, json.loads(body)))
        return reply("1")


class LongerTransport:
    """Pretends to be a judge that picks the longer response."""

    def __call__(self, url, headers, body, timeout):
        msg = json.loads(body)["messages"][1]["content"]
        first = msg.split("[Response 1]\n", 1)[1].split("\n\n[Response 2]")[0]
        second = msg.split("[Response 2]\n", 1)[1].split("\n\nWhich", 1)[0]
        return reply("2" if len(second) > len(first) else "1")


@pytest.mark.parametrize(
    "text,expected",
    [
        ("1", "first"),
        ("Response 2 is better.", "second"),
        ("tie", "tie"),
        ("I think A wins", "first"),
        ("b", "second"),
        ("The first one", "first"),
        ("", "tie"),
        ("no idea", "tie"),
    ],
)
def test_parse_verdict(text, expected):
    assert parse_verdict(text) == expected


def test_slot_to_name():
    assert slot_to_name("first", "AB") == "A"
    assert slot_to_name("second", "AB") == "B"
    assert slot_to_name("first", "BA") == "B"
    assert slot_to_name("second", "BA") == "A"
    assert slot_to_name("tie", "BA") == "tie"
    with pytest.raises(ValueError):
        slot_to_name("maybe", "AB")


def test_http_judge_builds_request_and_sends_key():
    transport = FirstSlotTransport()
    judge = HttpJudge("http://example.invalid/v1/", "m", api_key="k", transport=transport)
    assert judge("q", "x", "y") == "first"
    url, headers, body = transport.requests[0]
    assert url == "http://example.invalid/v1/chat/completions"
    assert headers["Authorization"] == "Bearer k"
    assert body["model"] == "m"
    assert body["messages"][0]["role"] == "system"
    assert "[Response 1]\nx" in body["messages"][1]["content"]
    assert judge.calls == 1


def test_http_judge_without_key_and_bad_reply():
    judge = HttpJudge("http://h/v1", "m", transport=lambda *a: {"error": "x"})
    with pytest.raises(RuntimeError):
        judge("q", "x", "y")
    seen = {}

    def t(url, headers, body, timeout):
        seen.update(headers)
        return reply("tie")

    assert HttpJudge("http://h/v1", "m", transport=t)("q", "x", "y") == "tie"
    assert "Authorization" not in seen


def test_run_pairs_first_slot_judge_shows_as_position_bias():
    pairs = [Pair(f"p{i}", "q", "a" * (10 + i), "b" * 12, "m1", "m2") for i in range(20)]
    judge = HttpJudge("http://h/v1", "m", transport=FirstSlotTransport())
    seen = []
    judgments = run_pairs(pairs, judge, judge_name="j", on_judgment=seen.append)
    assert len(judgments) == 40 == len(seen)
    # first-slot verdicts: A in AB rows, B in BA rows
    assert all(j.verdict == ("A" if j.order == "AB" else "B") for j in judgments)
    assert judgments[0].len_a == 10 and judgments[0].len_b == 12
    pos = position_bias(judgments, reps=50)
    assert pos["first_win_rate"]["value"] == 1.0
    assert pos["flip_rate"]["value"] == 1.0
    assert pos["first_both_rate"]["value"] == 1.0


def test_run_pairs_length_judge_has_no_position_bias():
    pairs = [Pair(f"p{i}", "q", "a" * (4 + 3 * i), "b" * 20) for i in range(12)]
    judge = HttpJudge("http://h/v1", "m", transport=LongerTransport())
    judgments = run_pairs(pairs, judge, orders=("AB", "BA"), repeats=2)
    assert len(judgments) == 48
    assert position_bias(judgments, reps=20)["flip_rate"]["value"] == 0.0
    longer = [j for j in judgments if j.len_a != j.len_b]
    assert all(j.verdict == ("A" if j.len_a > j.len_b else "B") for j in longer)


def test_pair_from_dict_and_load(tmp_path):
    with pytest.raises(ValueError):
        pair_from_dict({"pair_id": "x", "prompt": "q"})
    p = pair_from_dict({"pair_id": 1, "prompt": "q", "answer_a": "a", "answer_b": "b", "tag": 3})
    assert p.pair_id == "1" and p.extra == {"tag": 3} and p.model_a is None
    path = tmp_path / "pairs.jsonl"
    write_jsonl(path, [{"pair_id": "p", "prompt": "q", "answer_a": "a", "answer_b": "bb"}])
    assert load_pairs(str(path))[0].answer_b == "bb"


def test_cli_run_uses_transport(tmp_path, capsys, monkeypatch):
    pairs = tmp_path / "pairs.jsonl"
    write_jsonl(
        pairs,
        [{"pair_id": f"p{i}", "prompt": "q", "answer_a": "a", "answer_b": "bb"} for i in range(3)],
    )
    monkeypatch.setattr("judgebias.judge.urllib_transport", lambda *a: reply("2"))
    out = tmp_path / "judged.jsonl"
    rc = main(["run", str(pairs), "--out", str(out), "--model", "m", "--one-order"])
    assert rc == 0
    assert "wrote 3 judgments from 3 pairs" in capsys.readouterr().out
    rows = load_jsonl(out)
    assert [j.verdict for j in rows] == ["B", "B", "B"]
    assert rows[0].judge == "m" and rows[0].order == "AB"


def test_cli_run_reports_errors(tmp_path, capsys, monkeypatch):
    assert main(["run", str(tmp_path / "missing.jsonl"), "--out", "x", "--model", "m"]) == 2
    pairs = tmp_path / "pairs.jsonl"
    write_jsonl(pairs, [{"pair_id": "p", "prompt": "q", "answer_a": "a", "answer_b": "b"}])

    def boom(*a):
        raise RuntimeError("judge endpoint returned 500: down")

    monkeypatch.setattr("judgebias.judge.urllib_transport", boom)
    assert main(["run", str(pairs), "--out", str(tmp_path / "o"), "--model", "m"]) == 1
    assert "returned 500" in capsys.readouterr().err
