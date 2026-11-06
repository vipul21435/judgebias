"""Pairwise judgment records and JSONL loading.

A judgment is one verdict by a judge on a pair of answers to the same prompt.
``verdict`` is ``"A"``, ``"B"`` or ``"tie"`` and always refers to the answers as
named in the record, not to the order in which they were shown. ``order``
says which answer was shown first: ``"AB"`` (A first, the default) or ``"BA"``.

Some datasets only publish a verdict that already combines both presentation
orders, with a flag for pairs where the two orders disagreed. Such a row sets
``flipped`` (True if the orders disagreed, False if they agreed); its
``order`` then means nothing and the first-position measures skip it.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

VERDICTS = ("A", "B", "tie")
ORDERS = ("AB", "BA")


@dataclass(frozen=True)
class Judgment:
    pair_id: str
    verdict: str
    order: str = "AB"
    judge: str = "judge"
    len_a: int | None = None
    len_b: int | None = None
    model_a: str | None = None
    model_b: str | None = None
    human: str | None = None
    repeat: int = 0
    flipped: bool | None = None
    extra: dict[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        if self.verdict not in VERDICTS:
            raise ValueError(f"verdict must be one of {VERDICTS}, got {self.verdict!r}")
        if self.order not in ORDERS:
            raise ValueError(f"order must be one of {ORDERS}, got {self.order!r}")
        if self.human is not None and self.human not in VERDICTS:
            raise ValueError(f"human must be one of {VERDICTS}, got {self.human!r}")
        for name in ("len_a", "len_b"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or value < 0):
                raise ValueError(f"{name} must be a non-negative count, got {value!r}")

    @property
    def first_shown(self) -> str:
        """Name of the answer shown first."""
        return "A" if self.order == "AB" else "B"

    @property
    def picked_first(self) -> bool | None:
        """True if the verdict picked the first-shown answer.

        None on a tie and on rows whose verdict combines both orders.
        """
        if self.verdict == "tie" or self.flipped is not None:
            return None
        return self.verdict == self.first_shown


_ALIASES = {
    "pair_id": ("pair_id", "id", "question_id"),
    "verdict": ("verdict", "winner", "label"),
    "order": ("order",),
    "judge": ("judge",),
    "len_a": ("len_a", "length_a"),
    "len_b": ("len_b", "length_b"),
    "model_a": ("model_a",),
    "model_b": ("model_b",),
    "human": ("human", "human_label", "gold"),
    "repeat": ("repeat", "sample"),
    "flipped": ("flipped", "inconsistent"),
}


def _norm_verdict(value: Any) -> str:
    """Read a verdict by answer name.

    Numeric labels are refused: some datasets use 0 for A and 1 for B, while
    run mode and most judge prompts use 1 for the first answer, and a silent
    choice between the two would swap every verdict in a file. The slot words
    ``first`` and ``second`` are refused too: they name the position an answer
    was shown in, which is A or B depending on ``order``, and reading them as
    A and B would hide exactly the position bias the audit looks for.
    """
    text = str(value).strip().lower()
    if text in ("a", "model_a"):
        return "A"
    if text in ("b", "model_b"):
        return "B"
    if text in ("first", "second"):
        raise ValueError(
            f"slot word {value!r} names a position, not an answer; write A, B or tie"
        )
    if text in ("0", "1", "2", "0.0", "1.0", "2.0"):
        raise ValueError(
            f"numeric verdict {value!r} is ambiguous (0/1 or 1/2 for A/B?); "
            "write A, B or tie"
        )
    if text in ("tie", "draw", "equal", "both", "neither", "tie (bothbad)", "tie (bothgood)"):
        return "tie"
    raise ValueError(f"cannot read verdict {value!r}")


def _norm_flag(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("true", "1", "yes"):
        return True
    if text in ("false", "0", "no"):
        return False
    raise ValueError(f"cannot read flag {value!r}")


def _whole(name: str, value: Any) -> int:
    """Read a count field (a length or a repeat index) as a non-negative int."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a whole number, got {value!r}")
    if isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer():
            raise ValueError(f"{name} must be a whole number, got {value!r}")
        out = int(value)
    else:
        try:
            out = int(str(value).strip())
        except ValueError:
            raise ValueError(f"{name} must be a whole number, got {value!r}") from None
    if out < 0:
        raise ValueError(f"{name} must not be negative, got {value!r}")
    return out


def judgment_from_dict(row: dict[str, Any]) -> Judgment:
    """Build a Judgment from a dict, accepting a few common key spellings.

    If ``answer_a`` / ``answer_b`` text fields are present and no explicit
    lengths are, lengths are taken as character counts of the text. Judge and
    model names are read as text, so ``"judge": 5`` is the judge named "5".
    """
    if not isinstance(row, dict):
        raise ValueError(f"expected a JSON object, got {type(row).__name__}")
    kw: dict[str, Any] = {}
    used: set[str] = set()
    for name, keys in _ALIASES.items():
        for key in keys:
            if key in row and row[key] is not None:
                kw[name] = row[key]
                used.add(key)
                break
    if "pair_id" not in kw:
        raise ValueError("row has no pair_id")
    if "verdict" not in kw:
        raise ValueError("row has no verdict")
    kw["pair_id"] = str(kw["pair_id"])
    kw["verdict"] = _norm_verdict(kw["verdict"])
    if "human" in kw:
        kw["human"] = _norm_verdict(kw["human"])
    if "order" in kw:
        kw["order"] = str(kw["order"]).upper()
    if "flipped" in kw:
        kw["flipped"] = _norm_flag(kw["flipped"])
    for name in ("judge", "model_a", "model_b"):
        if name in kw:
            kw[name] = str(kw[name])
    for side in ("a", "b"):
        if f"len_{side}" not in kw and f"answer_{side}" in row:
            kw[f"len_{side}"] = len(str(row[f"answer_{side}"]))
            used.add(f"answer_{side}")
    for name in ("len_a", "len_b", "repeat"):
        if name in kw:
            kw[name] = _whole(name, kw[name])
    kw["extra"] = {k: v for k, v in row.items() if k not in used}
    return Judgment(**kw)


def iter_jsonl(path: str | Path) -> Iterator[tuple[int, Any]]:
    """Yield ``(line number, parsed value)`` for each non-blank line of a JSONL file.

    A UTF-8 byte order mark at the start of the file is skipped.
    """
    with Path(path).open(encoding="utf-8-sig") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield lineno, json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: bad JSON: {exc}") from exc


def load_jsonl(path: str | Path) -> list[Judgment]:
    """Read a JSONL file of judgments; a bad row raises ValueError naming its line."""
    out = []
    for lineno, row in iter_jsonl(path):
        try:
            out.append(judgment_from_dict(row))
        except (ValueError, TypeError, OverflowError) as exc:
            raise ValueError(f"{path}:{lineno}: {exc}") from exc
    return out


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    """Write one JSON object per line."""
    with Path(path).open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
