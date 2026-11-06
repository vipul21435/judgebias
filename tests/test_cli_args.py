import pytest

from judgebias.cli import main


@pytest.mark.parametrize(
    "argv",
    [
        ["audit", "x.jsonl", "--reps", "-5"],
        ["audit", "x.jsonl", "--reps", "0"],
        ["simulate", "x.jsonl", "--pairs", "-3"],
        ["simulate", "x.jsonl", "--repeats", "0"],
        ["run", "x.jsonl", "--out", "o", "--model", "m", "--repeats", "-1"],
        ["run", "x.jsonl", "--out", "o", "--model", "m", "--repeats", "0"],
        ["audit", "x.jsonl", "--reps", "many"],
    ],
)
def test_counts_must_be_positive(argv, capsys):
    with pytest.raises(SystemExit) as info:
        main(argv)
    assert info.value.code == 2
    assert "must be a whole number of 1 or more" in capsys.readouterr().err


def test_audit_json_is_strict_json(tmp_path, capsys):
    import json

    path = tmp_path / "two.jsonl"
    path.write_text(
        '{"pair_id": "p1", "verdict": "A", "human": "A"}\n'
        '{"pair_id": "p2", "verdict": "A", "human": "A"}\n'
    )
    assert main(["audit", str(path), "--json", "--reps", "5"]) == 0
    out = capsys.readouterr().out

    def refuse(token):
        raise ValueError(f"non-standard JSON token {token}")

    report = json.loads(out, parse_constant=refuse)
    assert report["position"]["flip_rate"]["value"] is None
    assert report["repeat"]["unanimous_rate"] is None
