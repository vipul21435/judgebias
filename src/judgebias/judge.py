"""Run mode: ask a judge for verdicts on pairs of answers.

A judge is any callable ``judge(prompt, first, second) -> verdict`` where
``verdict`` is ``"first"``, ``"second"`` or ``"tie"`` and refers to the slot
in which the answers were shown. :class:`HttpJudge` is one such callable that
talks to an OpenAI-compatible chat-completions endpoint through ``urllib``.
``run_pairs`` shows every pair in one or both orders, maps slot verdicts back
to answer names and returns :class:`Judgment` records ready for ``audit``.

Nothing in this module opens a network connection unless ``HttpJudge`` is
called with its default transport.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from .data import Judgment

Transport = Callable[[str, dict[str, str], bytes, float], dict[str, Any]]
JudgeFn = Callable[[str, str, str], str]

SLOT_VERDICTS = ("first", "second", "tie")

DEFAULT_SYSTEM = (
    "You compare two responses to the same user request and decide which one "
    "is better. Judge the content only. Reply with exactly one word: 1 if the "
    "first response is better, 2 if the second response is better, or tie."
)

DEFAULT_TEMPLATE = (
    "[Request]\n{prompt}\n\n[Response 1]\n{first}\n\n[Response 2]\n{second}\n\n"
    "Which response is better? Answer with 1, 2 or tie."
)


@dataclass(frozen=True)
class Pair:
    """One prompt with two candidate answers to compare."""

    pair_id: str
    prompt: str
    answer_a: str
    answer_b: str
    model_a: str | None = None
    model_b: str | None = None
    human: str | None = None
    extra: dict[str, Any] = field(default_factory=dict, compare=False)


def pair_from_dict(row: dict[str, Any]) -> Pair:
    known = ("pair_id", "prompt", "answer_a", "answer_b", "model_a", "model_b", "human")
    missing = [k for k in ("pair_id", "prompt", "answer_a", "answer_b") if k not in row]
    if missing:
        raise ValueError(f"pair row is missing {missing}")
    return Pair(
        pair_id=str(row["pair_id"]),
        prompt=str(row["prompt"]),
        answer_a=str(row["answer_a"]),
        answer_b=str(row["answer_b"]),
        model_a=row.get("model_a"),
        model_b=row.get("model_b"),
        human=row.get("human"),
        extra={k: v for k, v in row.items() if k not in known},
    )


def load_pairs(path: str) -> list[Pair]:
    from .data import iter_jsonl

    return [pair_from_dict(row) for row in iter_jsonl(path)]


_VERDICT_RE = re.compile(r"\b(1|2|tie|first|second|a|b)\b", re.IGNORECASE)


def parse_verdict(text: str) -> str:
    """Map a judge reply to ``first``, ``second`` or ``tie``.

    Takes the first token in the reply that reads as a verdict: ``1``,
    ``first`` or ``A`` mean the first slot, ``2``, ``second`` or ``B`` the
    second, ``tie`` a tie. Replies with no such token count as a tie, so an
    evasive judge shows up as ties rather than as a crash.
    """
    m = _VERDICT_RE.search(text)
    if not m:
        return "tie"
    tok = m.group(1).lower()
    if tok in ("1", "first", "a"):
        return "first"
    if tok in ("2", "second", "b"):
        return "second"
    return "tie"


def urllib_transport(
    url: str, headers: dict[str, str], body: bytes, timeout: float
) -> dict[str, Any]:
    """POST ``body`` to ``url`` and return the JSON reply."""
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:500]
        raise RuntimeError(f"judge endpoint returned {exc.code}: {detail}") from exc


@dataclass
class HttpJudge:
    """A judge backed by an OpenAI-compatible ``/chat/completions`` endpoint.

    ``base_url`` is the API root (``http://localhost:8000/v1``); the request
    goes to ``{base_url}/chat/completions``. ``api_key`` is sent as a bearer
    token when given. ``transport`` can be replaced to test without a server.
    """

    base_url: str
    model: str
    api_key: str | None = None
    system: str = DEFAULT_SYSTEM
    template: str = DEFAULT_TEMPLATE
    temperature: float = 0.0
    max_tokens: int = 8
    timeout: float = 60.0
    transport: Transport = urllib_transport
    calls: int = 0

    def request_body(self, prompt: str, first: str, second: str) -> dict[str, Any]:
        user = self.template.format(prompt=prompt, first=first, second=second)
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": self.system},
                {"role": "user", "content": user},
            ],
        }

    def __call__(self, prompt: str, first: str, second: str) -> str:
        url = self.base_url.rstrip("/") + "/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        body = json.dumps(self.request_body(prompt, first, second)).encode("utf-8")
        reply = self.transport(url, headers, body, self.timeout)
        self.calls += 1
        try:
            text = reply["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"unexpected reply shape: {json.dumps(reply)[:300]}") from exc
        return parse_verdict(str(text))


def slot_to_name(slot_verdict: str, order: str) -> str:
    """Translate a slot verdict into an answer name for the given order."""
    if slot_verdict == "tie":
        return "tie"
    if slot_verdict not in SLOT_VERDICTS:
        raise ValueError(f"judge returned {slot_verdict!r}, expected one of {SLOT_VERDICTS}")
    first_name = "A" if order == "AB" else "B"
    second_name = "B" if order == "AB" else "A"
    return first_name if slot_verdict == "first" else second_name


def run_pairs(
    pairs: Iterable[Pair],
    judge: JudgeFn,
    *,
    judge_name: str = "judge",
    orders: Sequence[str] = ("AB", "BA"),
    repeats: int = 1,
    on_judgment: Callable[[Judgment], None] | None = None,
) -> list[Judgment]:
    """Judge every pair in each order ``repeats`` times.

    ``judge`` sees the answers in slot order and returns a slot verdict; the
    returned :class:`Judgment` records carry verdicts by answer name, lengths
    in characters and whatever model names and human labels the pairs had.
    """
    out: list[Judgment] = []
    for p in pairs:
        for order in orders:
            first, second = (p.answer_a, p.answer_b) if order == "AB" else (p.answer_b, p.answer_a)
            for rep in range(repeats):
                verdict = slot_to_name(judge(p.prompt, first, second), order)
                j = Judgment(
                    pair_id=p.pair_id,
                    verdict=verdict,
                    order=order,
                    judge=judge_name,
                    len_a=len(p.answer_a),
                    len_b=len(p.answer_b),
                    model_a=p.model_a,
                    model_b=p.model_b,
                    human=p.human,
                    repeat=rep,
                )
                out.append(j)
                if on_judgment is not None:
                    on_judgment(j)
    return out
