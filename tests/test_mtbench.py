"""MT-Bench conversion on hand-built rows shaped like the parquet splits."""

import pytest

from judgebias.cli import main
from judgebias.data import judgment_from_dict, load_jsonl
from judgebias.measures import agreement, position_bias
from judgebias.mtbench import (
    answer_text,
    convert_human,
    convert_human_pairs,
    convert_judge,
    summary,
)


def conv(*answers):
    msgs = []
    for i, a in enumerate(answers):
        msgs.append({"role": "user", "content": f"question {i}"})
        msgs.append({"role": "assistant", "content": a})
    return msgs


def row(qid, ma, mb, winner, judge, turn=1, ans_a=("aa", "aaaa"), ans_b=("b", "bbb")):
    return {
        "question_id": qid,
        "model_a": ma,
        "model_b": mb,
        "winner": winner,
        "judge": judge,
        "conversation_a": conv(*ans_a),
        "conversation_b": conv(*ans_b),
        "turn": turn,
    }


def test_answer_text_picks_turn():
    c = conv("one", "two")
    assert answer_text(c, 1) == "one"
    assert answer_text(c, 2) == "two"
    with pytest.raises(ValueError):
        answer_text(c, 3)


def test_convert_human_canonical_order_and_verdict():
    rows = [
        row(1, "x-model", "y-model", "model_a", "expert_0"),
        row(1, "y-model", "x-model", "model_a", "expert_1"),  # shown swapped, picked y
        row(1, "y-model", "x-model", "tie", "expert_2", turn=2),
    ]
    out = convert_human(rows)
    assert [r["pair_id"] for r in out] == [
        "q1-t1-x-model-vs-y-model",
        "q1-t1-x-model-vs-y-model",
        "q1-t2-x-model-vs-y-model",
    ]
    assert [r["order"] for r in out] == ["AB", "BA", "BA"]
    assert [r["verdict"] for r in out] == ["A", "B", "tie"]
    assert out[0]["len_a"] == 2 and out[0]["len_b"] == 1
    # swapped rows: lengths follow the canonical names, x-model first
    assert out[1]["len_a"] == 1 and out[1]["len_b"] == 2
    assert out[2]["len_a"] == 3 and out[2]["len_b"] == 4
    assert out[1]["judge"] == "expert_1"


def test_convert_judge_pairs_with_humans_and_handles_inconsistent():
    judge_rows = [
        row(1, "x-model", "y-model", "model_b", "gpt4_pair"),
        row(2, "y-model", "x-model", "tie (inconsistent)", "gpt4_pair"),
        row(3, "x-model", "y-model", "tie", "gpt4_pair"),
    ]
    human_rows = [
        row(1, "y-model", "x-model", "model_a", "expert_0"),  # human picked y = canonical B
        row(1, "x-model", "y-model", "model_a", "expert_1"),  # human picked x = canonical A
        row(2, "x-model", "y-model", "model_b", "expert_0"),
        row(4, "x-model", "y-model", "model_b", "expert_0"),  # judge never saw q4
    ]
    out = convert_judge(judge_rows, human_rows)
    assert len(out) == 3
    assert out[0]["verdict"] == "B" and out[0]["human"] == "B"
    assert out[1]["verdict"] == "B" and out[1]["human"] == "A"
    assert out[2]["verdict"] == "tie" and out[2]["flipped"] is True
    assert out[0]["judge"] == "gpt-4" and out[0]["human_judge"] == "expert_0"
    dropped = convert_judge(judge_rows, human_rows, inconsistent="drop")
    assert len(dropped) == 2
    with pytest.raises(ValueError):
        convert_judge(judge_rows, human_rows, inconsistent="maybe")
    assert summary(judge_rows) == {"rows": 3, "decided": 1, "tie": 1, "inconsistent": 1}


def test_convert_human_pairs_baseline():
    rows = [
        row(1, "x-model", "y-model", "model_a", "expert_0"),
        row(1, "y-model", "x-model", "model_b", "expert_1"),  # also picked x
        row(1, "x-model", "y-model", "tie", "expert_2"),
        row(1, "x-model", "y-model", "model_b", "expert_0"),  # same judge again: no row
        row(2, "x-model", "y-model", "model_a", "expert_0"),  # only one judge: no row
    ]
    out = convert_human_pairs(rows)
    pairs = [(r["judge"], r["human_judge"]) for r in out]
    assert pairs == [
        ("expert_0", "expert_1"),
        ("expert_0", "expert_2"),
        ("expert_1", "expert_2"),
        ("expert_1", "expert_0"),
        ("expert_2", "expert_0"),
    ]
    assert out[0]["verdict"] == "A" and out[0]["human"] == "A"
    assert out[1]["human"] == "tie"
    assert out[2]["order"] == "BA"
    judgments = [judgment_from_dict(r) for r in out]
    assert agreement(judgments, reps=10)["raw"]["value"] == pytest.approx(1 / 5)


def test_converted_rows_load_and_audit(tmp_path):
    from judgebias.data import write_jsonl

    judge_rows = [row(q, "x-model", "y-model", "model_b", "gpt4_pair") for q in range(8)]
    human_rows = [
        row(q, "x-model", "y-model", "model_b" if q % 4 else "model_a", f"expert_{q % 2}")
        for q in range(8)
    ]
    path = tmp_path / "j.jsonl"
    write_jsonl(path, convert_judge(judge_rows, human_rows))
    judgments = load_jsonl(path)
    assert len(judgments) == 8
    agr = agreement(judgments, reps=20)
    assert agr["raw"]["value"] == pytest.approx(6 / 8)
    pos_judge = position_bias(judgments, reps=20)
    assert pos_judge["n_combined"] == 8 and pos_judge["n_decided"] == 0
    assert pos_judge["combined_flip_rate"]["value"] == 0.0
    hpath = tmp_path / "h.jsonl"
    write_jsonl(hpath, convert_human(human_rows))
    pos = position_bias(load_jsonl(hpath), reps=20)
    assert pos["first_win_rate"]["value"] == pytest.approx(2 / 8)


def test_cli_mtbench_offline(tmp_path, capsys, monkeypatch):
    data = {
        "gpt4_pair": [row(1, "x-model", "y-model", "model_a", "gpt4_pair")],
        "human": [row(1, "x-model", "y-model", "model_a", "expert_0")],
    }
    monkeypatch.setattr(
        "judgebias.mtbench.download",
        lambda d: {k: tmp_path / f"{k}.parquet" for k in data},
    )
    monkeypatch.setattr("judgebias.mtbench.read_parquet", lambda p: data[p.stem])
    humans = tmp_path / "h.jsonl"
    judge = tmp_path / "j.jsonl"
    pairs = tmp_path / "hh.jsonl"
    argv = ["mtbench", "--data-dir", str(tmp_path), "--humans", str(humans), "--judge", str(judge)]
    argv += ["--human-pairs", str(pairs)]
    rc = main(argv)
    assert rc == 0
    out = capsys.readouterr().out
    assert "human rows 1; judge rows 1" in out
    assert load_jsonl(humans)[0].judge == "expert_0"
    assert load_jsonl(judge)[0].human == "A"
    assert "wrote 0 human-human rows" in out


def test_download_and_read_parquet(tmp_path, monkeypatch):
    pq = pytest.importorskip("pyarrow.parquet")
    import io

    import pyarrow as pa

    from judgebias.mtbench import download, read_parquet

    table = pa.table({"question_id": [1, 2], "winner": ["model_a", "tie"]})
    buf = io.BytesIO()
    pq.write_table(table, buf)
    payload = buf.getvalue()

    class FakeResponse(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    urls = []

    def fake_urlopen(url, timeout=0):
        urls.append(url)
        return FakeResponse(payload)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    paths = download(tmp_path / "d")
    assert sorted(paths) == ["gpt4_pair", "human"]
    assert len(urls) == 2 and all(u.startswith("https://huggingface.co/") for u in urls)
    assert read_parquet(paths["human"]) == [
        {"question_id": 1, "winner": "model_a"},
        {"question_id": 2, "winner": "tie"},
    ]
    download(tmp_path / "d")
    assert len(urls) == 2  # cached, nothing fetched again


def test_cli_mtbench_turn_filter(tmp_path, capsys, monkeypatch):
    data = {
        "gpt4_pair": [row(1, "x-model", "y-model", "model_a", "gpt4_pair", turn=t) for t in (1, 2)],
        "human": [row(1, "x-model", "y-model", "model_a", "expert_0", turn=t) for t in (1, 2)],
    }
    monkeypatch.setattr(
        "judgebias.mtbench.download",
        lambda d: {k: tmp_path / f"{k}.parquet" for k in data},
    )
    monkeypatch.setattr("judgebias.mtbench.read_parquet", lambda p: data[p.stem])
    judge = tmp_path / "j.jsonl"
    assert main(["mtbench", "--data-dir", str(tmp_path), "--judge", str(judge), "--turn", "2"]) == 0
    assert "human rows 1; judge rows 1" in capsys.readouterr().out
    assert [x.extra["turn"] for x in load_jsonl(judge)] == [2]
    data["human"].append(row(1, "x-model", "y-model", "model_b", "author_0", turn=1))
    argv = ["mtbench", "--data-dir", str(tmp_path), "--judge", str(judge)]
    assert main(argv + ["--human-prefix", "expert"]) == 0
    assert "human rows 2; judge rows 2" in capsys.readouterr().out
    assert {x.extra["human_judge"] for x in load_jsonl(judge)} == {"expert_0"}


def test_cli_mtbench_reports_download_and_parquet_errors(tmp_path, capsys, monkeypatch):
    import urllib.error

    def offline(url, timeout):
        raise urllib.error.URLError("name resolution failed")

    monkeypatch.setattr("urllib.request.urlopen", offline)
    assert main(["mtbench", "--data-dir", str(tmp_path / "d")]) == 2
    assert "name resolution failed" in capsys.readouterr().err

    monkeypatch.setattr("judgebias.mtbench.download", lambda d: {"human": "h", "gpt4_pair": "g"})

    def no_pyarrow(path):
        raise RuntimeError("reading parquet needs pyarrow: pip install 'judgebias[mtbench]'")

    monkeypatch.setattr("judgebias.mtbench.read_parquet", no_pyarrow)
    assert main(["mtbench", "--data-dir", str(tmp_path / "d")]) == 2
    err = capsys.readouterr().err
    assert err.startswith("judgebias: reading parquet needs pyarrow") and "Traceback" not in err


def test_missing_pyarrow_hint_points_at_this_repository(monkeypatch):
    import sys

    from judgebias.mtbench import read_parquet

    monkeypatch.setitem(sys.modules, "pyarrow", None)
    monkeypatch.setitem(sys.modules, "pyarrow.parquet", None)
    with pytest.raises(RuntimeError) as info:
        read_parquet("x.parquet")
    msg = str(info.value)
    assert "git+https://github.com/vipul21435/judgebias" in msg
    assert "pip install 'judgebias[mtbench]'" not in msg

