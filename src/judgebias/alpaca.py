"""Convert the public AlpacaFarm and AlpacaEval preference files into judgebias JSONL.

Two sources, both JSON lists of rows with ``instruction``, ``output_1``,
``output_2`` and ``preference``:

- AlpacaFarm (``tatsu-lab/alpaca_farm`` on Hugging Face): one human preference
  and, in a separate file, one GPT-4 preference per pair of outputs, plus a
  cross-annotation file (in ``tatsu-lab/alpaca_eval``) in which four human
  annotators, named by ``annotator_index``, labelled each of about 650 pairs.
- AlpacaEval (``tatsu-lab/alpaca_eval``): verdicts of a GPT-4 judge and a
  Claude judge on the same 10,465 pairs, each an answer by one of 13 models
  (``generator_2``) against the ``text_davinci_003`` baseline
  (``generator_1``). These files carry no human labels.

``preference`` is ``1`` (output_1 preferred), ``2`` (output_2) or ``1.5``
(tie), written as text or as a number. ``0`` marks identical outputs, which
the judge was not asked about, and a missing value marks a reply that could
not be read; both are left out and counted.

The files record ``output_1`` and ``output_2`` in a fixed order but not the
order in which a judge saw them, so ``order`` here is the stored order. The
first-position measures of an audit then describe ``output_1``, not the
first slot; see the README.

Pairs carry no ids in these files. Rows are matched on their texts and given
sequential ids (``af00001`` and so on) in the order the pairs first appear.
"""

from __future__ import annotations

import json
import urllib.request
from collections.abc import Iterable
from pathlib import Path
from typing import Any

_HF = "https://huggingface.co/datasets/tatsu-lab"
URLS = {
    "farm_human": f"{_HF}/alpaca_farm/resolve/main/alpaca_human_preference.json",
    "farm_gpt4": f"{_HF}/alpaca_farm/resolve/main/alpaca_gpt4_preference.json",
    "farm_cross": f"{_HF}/alpaca_eval/resolve/main/alpaca_farm_human_crossannotations.json",
    "eval_gpt4": f"{_HF}/alpaca_eval/resolve/main/alpaca_eval_annotations_alpaca_eval_gpt4.json",
    "eval_claude": f"{_HF}/alpaca_eval/resolve/main/alpaca_eval_annotations_claude.json",
}
FARM_JUDGE = "gpt-4"


def download(dest_dir: str | Path, names: Iterable[str]) -> dict[str, Path]:
    """Fetch the named files of ``URLS`` into ``dest_dir`` unless already present."""
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}
    for name in names:
        path = dest / f"{name}.json"
        if not path.exists():
            part = path.with_name(path.name + ".part")
            with urllib.request.urlopen(URLS[name], timeout=300) as resp:  # noqa: S310
                part.write_bytes(resp.read())
            part.replace(path)
        out[name] = path
    return out


def read_json(path: str | Path) -> list[dict[str, Any]]:
    """Read one of the JSON list files."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a JSON list of rows")
    return data


def preference(value: Any) -> str | None:
    """Read a ``preference`` value as ``A`` / ``B`` / ``tie``, or None if not judged."""
    if value is None:
        return None
    try:
        num = float(str(value).strip())
    except ValueError:
        if str(value).strip().lower() in ("", "none", "nan"):
            return None
        raise ValueError(f"cannot read preference {value!r}") from None
    if num == 1.0:
        return "A"
    if num == 2.0:
        return "B"
    if num == 1.5:
        return "tie"
    if num == 0.0 or num != num:  # identical outputs, or NaN for an unread reply
        return None
    raise ValueError(f"cannot read preference {value!r}")


def _swap(verdict: str) -> str:
    return {"A": "B", "B": "A", "tie": "tie"}[verdict]


def _key(row: dict[str, Any], with_input: bool = True) -> tuple[str, ...]:
    head = (str(row["instruction"]),)
    if with_input:
        head += (str(row.get("input") or ""),)
    return head + (str(row["output_1"]), str(row["output_2"]))


def _flip_key(key: tuple[str, ...]) -> tuple[str, ...]:
    return key[:-2] + (key[-1], key[-2])


class _Ids:
    """Sequential pair ids, handed out in order of first appearance."""

    def __init__(self, prefix: str, width: int = 5) -> None:
        self.prefix, self.width, self.seen = prefix, width, {}

    def __call__(self, key: Any) -> str:
        if key not in self.seen:
            self.seen[key] = f"{self.prefix}{len(self.seen) + 1:0{self.width}d}"
        return self.seen[key]


def convert_farm(
    judge_rows: Iterable[dict[str, Any]],
    human_rows: Iterable[dict[str, Any]],
    *,
    judge_name: str = FARM_JUDGE,
) -> list[dict[str, Any]]:
    """One row per AlpacaFarm human preference, carrying the GPT-4 verdict on the same pair.

    A and B are ``output_1`` and ``output_2`` of the human file. A GPT-4 row
    stored with the two outputs the other way round is matched too, with
    ``order`` ``BA`` and its verdict turned into the human file's names; a
    pair found both ways gives two rows. Pairs with no readable GPT-4 or
    human verdict are skipped.
    """
    by_key: dict[tuple[str, ...], str] = {}
    for row in judge_rows:
        v = preference(row.get("preference"))
        if v is not None:
            by_key[_key(row)] = v
    ids = _Ids("af")
    out = []
    for row in human_rows:
        human = preference(row.get("preference"))
        if human is None:
            continue
        key = _key(row)
        found = []
        if key in by_key:
            found.append(("AB", by_key[key]))
        if _flip_key(key) in by_key and key[-1] != key[-2]:
            found.append(("BA", _swap(by_key[_flip_key(key)])))
        for order, verdict in found:
            out.append(
                {
                    "pair_id": ids(key),
                    "verdict": verdict,
                    "order": order,
                    "judge": judge_name,
                    "len_a": len(key[-2]),
                    "len_b": len(key[-1]),
                    "human": human,
                }
            )
    return out


def _cross_votes(cross_rows: Iterable[dict[str, Any]]) -> dict[tuple[str, ...], list]:
    votes: dict[tuple[str, ...], list[tuple[str, str]]] = {}
    for row in cross_rows:
        v = preference(row.get("preference"))
        if v is None:
            continue
        name = f"annotator_{row['annotator_index']}"
        votes.setdefault(_key(row, with_input=False), []).append((name, v))
    return votes


def convert_cross_pairs(cross_rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Human-against-human rows from the cross-annotations.

    For each pair, every two different annotators give one row: the earlier
    one's label as ``verdict`` (with its name as ``judge``) and the later
    one's as ``human``.
    """
    ids = _Ids("ax", 4)
    out = []
    for key, votes in _cross_votes(cross_rows).items():
        for i in range(len(votes)):
            for k in range(i + 1, len(votes)):
                (ja, va), (jb, vb) = votes[i], votes[k]
                if ja == jb:
                    continue
                out.append(
                    {
                        "pair_id": ids(key),
                        "verdict": va,
                        "judge": ja,
                        "len_a": len(key[-2]),
                        "len_b": len(key[-1]),
                        "human": vb,
                        "human_judge": jb,
                    }
                )
    return out


def convert_cross_judge(
    judge_rows: Iterable[dict[str, Any]],
    cross_rows: Iterable[dict[str, Any]],
    *,
    judge_name: str = FARM_JUDGE,
) -> list[dict[str, Any]]:
    """The GPT-4 verdict against each cross-annotator, on pairs both labelled.

    Matching uses the instruction and the two outputs in the stored order
    (the cross-annotation file has no ``input`` field).
    """
    by_key: dict[tuple[str, ...], str] = {}
    for row in judge_rows:
        v = preference(row.get("preference"))
        if v is not None:
            by_key[_key(row, with_input=False)] = v
    ids = _Ids("ax", 4)
    out = []
    for key, votes in _cross_votes(cross_rows).items():
        if key not in by_key:
            continue
        for name, v in votes:
            out.append(
                {
                    "pair_id": ids(key),
                    "verdict": by_key[key],
                    "judge": judge_name,
                    "len_a": len(key[-2]),
                    "len_b": len(key[-1]),
                    "human": v,
                    "human_judge": name,
                }
            )
    return out


def _eval_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (str(row["instruction"]), str(row["generator_1"]), str(row["generator_2"]))


def convert_eval(*judge_files: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """AlpacaEval verdicts of one or more judges, one row per readable verdict.

    The judge is named by each row's ``annotator``; A is ``generator_1`` (the
    baseline) and B is ``generator_2``. Returns the rows and the number of
    rows left out (identical outputs or an unread reply).
    """
    ids = _Ids("ae")
    out, skipped = [], 0
    for rows in judge_files:
        for row in rows:
            v = preference(row.get("preference"))
            if v is None:
                skipped += 1
                continue
            out.append(
                {
                    "pair_id": ids(_eval_key(row)),
                    "verdict": v,
                    "judge": str(row["annotator"]),
                    "len_a": len(str(row["output_1"])),
                    "len_b": len(str(row["output_2"])),
                    "model_a": str(row["generator_1"]),
                    "model_b": str(row["generator_2"]),
                }
            )
    return out, skipped


def convert_eval_cross(
    rows_x: Iterable[dict[str, Any]], rows_y: Iterable[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Judge X's verdict with judge Y's in the ``human`` field, on pairs both read.

    Auditing the result gives the agreement of X with Y, and the
    self-preference excess becomes X's own-family win rate minus Y's on the
    same rows.
    """
    y = {}
    for row in rows_y:
        v = preference(row.get("preference"))
        if v is not None:
            y[_eval_key(row)] = (v, str(row["annotator"]))
    ids = _Ids("ae")
    out = []
    for row in rows_x:
        v = preference(row.get("preference"))
        key = _eval_key(row)
        if v is None or key not in y:
            continue
        out.append(
            {
                "pair_id": ids(key),
                "verdict": v,
                "judge": str(row["annotator"]),
                "len_a": len(str(row["output_1"])),
                "len_b": len(str(row["output_2"])),
                "model_a": key[1],
                "model_b": key[2],
                "human": y[key][0],
                "human_judge": y[key][1],
            }
        )
    return out
