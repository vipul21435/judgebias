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
