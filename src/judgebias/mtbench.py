"""Convert the public MT-Bench human-judgment data into judgebias JSONL.

The dataset ``lmsys/mt_bench_human_judgments`` on Hugging Face has two
splits. ``human`` holds one row per human verdict on a pair of answers
(``model_a``, ``model_b``, ``winner``, ``judge``, the two conversations and
the turn judged). ``gpt4_pair`` holds one row per unordered pair with the
verdict of a GPT-4 judge that saw both presentation orders; a verdict of
``tie (inconsistent)`` means the two orders disagreed.

``convert_human`` turns the human split into judgments by the human judges,
with the displayed order kept, so the position and length measures apply to
the humans themselves. ``convert_judge`` pairs every human verdict with the
GPT-4 verdict on the same pair, so the agreement and self-preference measures
apply to the GPT-4 judge against human labels.

Reading parquet needs ``pyarrow`` (install the ``mtbench`` extra); nothing
else here depends on it.
"""

from __future__ import annotations

import urllib.request
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

BASE = "https://huggingface.co/api/datasets/lmsys/mt_bench_human_judgments/parquet/default"
URLS = {"human": f"{BASE}/human/0.parquet", "gpt4_pair": f"{BASE}/gpt4_pair/0.parquet"}
DATASET_PAGE = "https://huggingface.co/datasets/lmsys/mt_bench_human_judgments"
JUDGE_NAME = "gpt-4"


def download(dest_dir: str | Path) -> dict[str, Path]:
    """Fetch both parquet splits into ``dest_dir`` unless already present."""
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}
    for split, url in URLS.items():
        path = dest / f"{split}.parquet"
        if not path.exists():
            with urllib.request.urlopen(url, timeout=120) as resp:  # noqa: S310
                path.write_bytes(resp.read())
        out[split] = path
    return out


def read_parquet(path: str | Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - depends on the environment
        msg = "reading parquet needs pyarrow: pip install 'judgebias[mtbench]'"
        raise RuntimeError(msg) from exc
    return pq.read_table(str(path)).to_pylist()


def answer_text(conversation: Sequence[dict[str, Any]], turn: int) -> str:
    """The assistant message judged at ``turn`` (1-based) of a conversation."""
    assistant = [m["content"] for m in conversation if m.get("role") == "assistant"]
    if turn < 1 or turn > len(assistant):
        raise ValueError(f"conversation has {len(assistant)} assistant turns, asked for {turn}")
    return str(assistant[turn - 1])


def _canonical(row: dict[str, Any]) -> dict[str, Any]:
    """Name the pair by its sorted model names and express the verdict in those names.

    ``order`` records which of the canonical A/B was shown first in the row.
    """
    ma, mb = str(row["model_a"]), str(row["model_b"])
    turn = int(row["turn"])
    swapped = ma > mb
    a, b = (mb, ma) if swapped else (ma, mb)
    winner = str(row["winner"])
    inconsistent = winner == "tie (inconsistent)"
    if winner == "model_a":
        verdict = "B" if swapped else "A"
    elif winner == "model_b":
        verdict = "A" if swapped else "B"
    else:
        verdict = "tie"
    text_a = answer_text(row["conversation_a"], turn)
    text_b = answer_text(row["conversation_b"], turn)
    len_a, len_b = (len(text_b), len(text_a)) if swapped else (len(text_a), len(text_b))
    return {
        "pair_id": f"q{row['question_id']}-t{turn}-{a}-vs-{b}",
        "question_id": int(row["question_id"]),
        "turn": turn,
        "order": "BA" if swapped else "AB",
        "model_a": a,
        "model_b": b,
        "verdict": verdict,
        "inconsistent": inconsistent,
        "len_a": len_a,
        "len_b": len_b,
    }


def convert_human(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """One judgment per human verdict, keeping the displayed order."""
    out = []
    for row in rows:
        c = _canonical(row)
        out.append(
            {
                "pair_id": c["pair_id"],
                "verdict": c["verdict"],
                "order": c["order"],
                "judge": str(row["judge"]),
                "len_a": c["len_a"],
                "len_b": c["len_b"],
                "model_a": c["model_a"],
                "model_b": c["model_b"],
                "question_id": c["question_id"],
                "turn": c["turn"],
            }
        )
    return out


def convert_judge(
    judge_rows: Iterable[dict[str, Any]],
    human_rows: Iterable[dict[str, Any]],
    *,
    inconsistent: str = "tie",
    judge_name: str = JUDGE_NAME,
) -> list[dict[str, Any]]:
    """One judgment per human verdict, carrying the GPT-4 verdict on the same pair.

    ``inconsistent="tie"`` keeps order-inconsistent GPT-4 verdicts as ties and
    marks them; ``"drop"`` leaves those pairs out. Human verdicts on pairs the
    judge never saw are skipped.
    """
    if inconsistent not in ("tie", "drop"):
        raise ValueError("inconsistent must be 'tie' or 'drop'")
    by_pair: dict[str, dict[str, Any]] = {}
    for row in judge_rows:
        c = _canonical(row)
        by_pair[c["pair_id"]] = c
    out = []
    for row in human_rows:
        h = _canonical(row)
        j = by_pair.get(h["pair_id"])
        if j is None or (inconsistent == "drop" and j["inconsistent"]):
            continue
        out.append(
            {
                "pair_id": h["pair_id"],
                "verdict": j["verdict"],
                "order": "AB",
                "judge": judge_name,
                "len_a": j["len_a"],
                "len_b": j["len_b"],
                "model_a": j["model_a"],
                "model_b": j["model_b"],
                "human": h["verdict"],
                "human_judge": str(row["judge"]),
                "inconsistent": j["inconsistent"],
                "question_id": h["question_id"],
                "turn": h["turn"],
            }
        )
    return out


def summary(judge_rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    """Counts of the GPT-4 split's verdict kinds, including order-inconsistent ones."""
    counts = {"rows": 0, "decided": 0, "tie": 0, "inconsistent": 0}
    for row in judge_rows:
        counts["rows"] += 1
        w = str(row["winner"])
        if w == "tie (inconsistent)":
            counts["inconsistent"] += 1
        elif w == "tie":
            counts["tie"] += 1
        else:
            counts["decided"] += 1
    return counts
