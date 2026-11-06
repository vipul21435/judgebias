import pytest

from judgebias import Judgment, audit, self_preference
from judgebias.cli import main, render
from judgebias.measures import family


def _rows(judge, a, b, n=40):
    # the judge always picks the answer from model a
    return [
        Judgment(pair_id=f"p{i}", verdict="A", judge=judge, model_a=a, model_b=b)
        for i in range(n)
    ]


def test_family_ignores_vendor_and_router_prefixes():
    assert family("openai/gpt-4o") == "gpt"
    assert family("meta-llama/llama-3-8b") == "llama"
    assert family("openrouter/meta-llama/llama-3-70b") == "llama"
    assert family("org/alpha_7b:chat") == "alpha"


def test_prefixed_judge_name_keeps_the_self_preference_section():
    rep = audit(_rows("openai/gpt-4o", "gpt-4o-mini", "claude-3-haiku"), reps=20)
    sp = rep["self_preference"]
    assert sp["judge_family"] == "gpt"
    assert sp["n_used"] == 40
    assert sp["own_win_rate"]["value"] == 1.0


def test_family_override_map():
    js = _rows("house-judge", "acme-small", "other-7b")
    assert "self_preference" not in audit(js, reps=5)
    sp = self_preference(js, reps=5, families={"house-judge": "acme"})
    assert sp["judge_family"] == "acme"
    assert sp["n_used"] == 40
    assert family("House-Judge", {"house-judge": "Acme"}) == "acme"


def test_audit_notes_when_no_row_pits_the_judge_family(tmp_path, capsys):
    rep = audit(_rows("house-judge", "acme-small", "other-7b"), reps=5)
    note = rep["self_preference_note"]
    assert note == {"judge_families": ["house"], "candidate_families": ["acme", "other"]}
    text = render(rep)
    assert "self-preference: n/a" in text
    assert "judge family house" in text and "acme, other" in text


def test_cli_family_flag(tmp_path, capsys):
    from judgebias.data import write_jsonl
    from judgebias.synth import to_rows

    path = tmp_path / "f.jsonl"
    write_jsonl(path, to_rows(_rows("house-judge", "acme-small", "other-7b", n=10)))
    assert main(["audit", str(path), "--reps", "5", "--family", "house-judge=acme"]) == 0
    out = capsys.readouterr().out
    assert "self-preference (judge family acme)" in out
    assert "own-family win rate      1.000" in out
    with pytest.raises(SystemExit):
        main(["audit", str(path), "--family", "no-equals-sign"])

