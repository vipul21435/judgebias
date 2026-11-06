import json

from judgebias import Judgment, audit, repeat_consistency
from judgebias.cli import main
from judgebias.measures import conflicting_copies


def _samples_without_repeat():
    # three samples per pair and order, verdicts differ, no repeat field
    out = []
    for i in range(10):
        for order in ("AB", "BA"):
            for v in ("A", "B", "A"):
                out.append(Judgment(pair_id=f"p{i}", verdict=v, order=order, judge="j"))
    return out


def test_conflicting_copies_are_counted():
    js = _samples_without_repeat()
    c = conflicting_copies(js)
    assert c == {"n_keys": 20, "n_rows": 40}
    plain = [Judgment(pair_id="p", verdict="A"), Judgment(pair_id="p", verdict="A")]
    assert conflicting_copies(plain) == {"n_keys": 0, "n_rows": 0}


def test_audit_and_repeat_report_conflicting_copies():
    js = _samples_without_repeat()
    rep = audit(js, reps=10)
    assert rep["conflicting_copies"] == {"n_keys": 20, "n_rows": 40}
    assert repeat_consistency(js)["n_conflicting_keys"] == 20


def test_cli_warns_on_conflicting_copies(tmp_path, capsys):
    path = tmp_path / "s.jsonl"
    rows = [{"pair_id": j.pair_id, "verdict": j.verdict, "order": j.order, "judge": j.judge}
            for j in _samples_without_repeat()]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    assert main(["audit", str(path), "--reps", "10"]) == 0
    captured = capsys.readouterr()
    assert "differ in verdict" in captured.err
    assert "add a repeat field" in captured.out
