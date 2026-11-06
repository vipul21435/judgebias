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
