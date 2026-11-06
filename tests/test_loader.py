import json

import pytest

from judgebias import judgment_from_dict
from judgebias.cli import main


def _write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


@pytest.mark.parametrize("word", ["first", "second", "First", " SECOND "])
def test_slot_words_are_refused(word):
    with pytest.raises(ValueError, match="names a position"):
        judgment_from_dict({"pair_id": "p1", "verdict": word, "order": "BA"})
    with pytest.raises(ValueError, match="names a position"):
        judgment_from_dict({"pair_id": "p1", "verdict": "A", "human": word})


def test_slot_word_file_is_not_read_as_unbiased(tmp_path, capsys):
    # a judge that always takes the first slot, written with slot words
    rows = []
    for i in range(20):
        rows.append({"pair_id": f"p{i}", "verdict": "first", "order": "AB"})
        rows.append({"pair_id": f"p{i}", "verdict": "first", "order": "BA"})
    path = tmp_path / "slot.jsonl"
    _write(path, rows)
    assert main(["audit", str(path), "--reps", "20"]) == 2
    err = capsys.readouterr().err
    assert "names a position" in err


def test_slot_word_error_names_the_line(tmp_path, capsys):
    path = tmp_path / "slot.jsonl"
    _write(path, [{"pair_id": "p1", "verdict": "A"}, {"pair_id": "p1", "verdict": "first"}])
    assert main(["audit", str(path)]) == 2
    assert "slot.jsonl:2:" in capsys.readouterr().err


def test_row_without_verdict_is_a_one_line_error(tmp_path, capsys):
    # a run-mode pairs file passed to audit by mistake
    path = tmp_path / "pairs.jsonl"
    _write(path, [{"pair_id": "p1", "prompt": "q", "answer_a": "x", "answer_b": "y"}])
    assert main(["audit", str(path)]) == 2
    err = capsys.readouterr().err
    assert "pairs.jsonl:1: row has no verdict" in err


@pytest.mark.parametrize("line", ["42", '"text"', "[1, 2]", "null"])
def test_non_object_line_is_refused(tmp_path, capsys, line):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"pair_id": "p1", "verdict": "A"}\n' + line + "\n", encoding="utf-8")
    assert main(["audit", str(path)]) == 2
    err = capsys.readouterr().err
    assert "bad.jsonl:2:" in err and "JSON object" in err


def test_numeric_names_are_read_as_text(tmp_path, capsys):
    rows = [
        {"pair_id": "p1", "verdict": "A", "judge": 5, "model_a": 5, "model_b": "b-1"},
        {"pair_id": "p2", "verdict": "B", "judge": "x", "model_a": "a-1", "model_b": 7},
    ]
    path = tmp_path / "num.jsonl"
    _write(path, rows)
    assert main(["audit", str(path), "--reps", "10"]) == 0
    out = capsys.readouterr().out
    assert "judges: 5, x" in out
    j = judgment_from_dict(rows[0])
    assert (j.judge, j.model_a, j.model_b) == ("5", "5", "b-1")


@pytest.mark.parametrize("value", ["Infinity", "NaN", "1e400"])
def test_infinite_length_is_a_one_line_error(tmp_path, capsys, value):
    path = tmp_path / "inf.jsonl"
    line = '{"pair_id": "p1", "verdict": "A", "len_a": ' + value + ', "len_b": 3}\n'
    path.write_text(line, encoding="utf-8")
    assert main(["audit", str(path)]) == 2
    assert "inf.jsonl:1:" in capsys.readouterr().err


def test_bad_verdict_and_repeat_errors_name_the_line(tmp_path, capsys):
    path = tmp_path / "bad.jsonl"
    _write(path, [{"pair_id": "p1", "verdict": "A"}, {"pair_id": "p2", "verdict": "X"}])
    assert main(["audit", str(path)]) == 2
    assert "bad.jsonl:2: cannot read verdict 'X'" in capsys.readouterr().err
    _write(path, [{"pair_id": "p1", "verdict": "A", "repeat": "3.5"}])
    assert main(["audit", str(path)]) == 2
    assert "bad.jsonl:1:" in capsys.readouterr().err


def test_utf8_bom_is_accepted(tmp_path):
    from judgebias import load_jsonl

    path = tmp_path / "bom.jsonl"
    path.write_bytes(b"\xef\xbb\xbf" + b'{"pair_id": "p1", "verdict": "A"}\n')
    assert [j.verdict for j in load_jsonl(path)] == ["A"]


@pytest.mark.parametrize("value", [3.9, True, -2, "-2", "abc"])
def test_lengths_must_be_non_negative_whole_numbers(value):
    with pytest.raises(ValueError, match="len_a"):
        judgment_from_dict({"pair_id": "p1", "verdict": "A", "len_a": value, "len_b": 3})


def test_whole_number_lengths_in_other_spellings_are_read():
    j = judgment_from_dict({"pair_id": "p1", "verdict": "A", "len_a": 4.0, "len_b": "7"})
    assert (j.len_a, j.len_b) == (4, 7)


def test_label_is_not_read_as_the_judge_verdict():
    # many preference datasets use "label" for the human label
    with pytest.raises(ValueError, match="no verdict"):
        judgment_from_dict({"pair_id": "p1", "label": "A"})
    j = judgment_from_dict({"pair_id": "p1", "verdict": "B", "label": "A"})
    assert j.verdict == "B" and j.extra == {"label": "A"}
