"""Command line entry point: audit, simulate, run and mtbench."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Sequence
from typing import Any

from . import __version__
from .data import load_jsonl, write_jsonl
from .judge import HttpJudge, load_pairs, run_pairs
from .measures import audit
from .synth import simulate, to_rows


def _fmt(est: dict[str, Any], signed: bool = False) -> str:
    v = est["value"]
    if isinstance(v, float) and math.isnan(v):
        return "n/a"
    f = "+.3f" if signed else ".3f"
    return f"{v:{f}} [{est['low']:{f}}, {est['high']:{f}}] n={est['n']}"


def render(report: dict[str, Any]) -> str:
    judges = report["judges"]
    names = ", ".join(judges[:5]) + (f", ... ({len(judges)} in all)" if len(judges) > 5 else "")
    lines = [f"judgments: {report['n_judgments']}  judges: {names}"]
    p = report["position"]
    lines.append("position bias")
    if p["n_decided"] + p["n_ties"] or not p.get("n_combined"):
        lines.append(
            f"  first-position win rate  {_fmt(p['first_win_rate'])}"
            f"  (decided {p['n_decided']}, ties {p['n_ties']})"
        )
        lines.append(
            f"  flip rate under swap     {_fmt(p['flip_rate'])}"
            f"  (pairs with both orders {p['n_pairs_both_orders']})"
        )
        lines.append(f"  first-both among flips   {_fmt(p['first_both_rate'])}")
    if "combined_flip_rate" in p:
        lines.append(
            f"  flipped (combined-order) {_fmt(p['combined_flip_rate'])}"
            f"  (combined-order verdicts {p['n_combined']})"
        )
    if "length" in report:
        ln = report["length"]
        lines.append("length bias")
        lines.append(f"  longer-answer win rate   {_fmt(ln['longer_win_rate'])}")
        lines.append(f"  logistic slope (log len ratio)  {_fmt(ln['slope'])}")
        if "slope_given_human" in ln:
            lines.append(f"  same, given human label  {_fmt(ln['slope_given_human'])}")
    if "self_preference" in report:
        s = report["self_preference"]
        lines.append(f"self-preference (judge family {s['judge_family']})")
        lines.append(f"  own-family win rate      {_fmt(s['own_win_rate'])}")
        if "own_win_rate_human" in s:
            lines.append(f"  same, by human labels    {_fmt(s['own_win_rate_human'])}")
            lines.append(f"  excess                   {_fmt(s['excess'], signed=True)}")
    if "agreement" in report:
        a = report["agreement"]
        lines.append(f"agreement with humans (ties={a['ties']}, labelled {a['n_labelled']})")
        lines.append(f"  raw agreement            {_fmt(a['raw'])}")
        lines.append(f"  Cohen's kappa            {_fmt(a['kappa'])}")
    r = report["repeat"]
    if r["n_groups"]:
        lines.append("repeat consistency")
        lines.append(
            f"  unanimous groups         {r['unanimous_rate']:.3f}  (groups {r['n_groups']})"
        )
        lines.append(f"  pairwise agreement       {r['pairwise_agreement']:.3f}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="judgebias", description=__doc__)
    parser.add_argument("--version", action="version", version=f"judgebias {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    a = sub.add_parser("audit", help="measure biases in a JSONL file of pairwise judgments")
    a.add_argument("path", help="JSONL file; one judgment per line")
    a.add_argument("--ties", choices=("keep", "drop"), default="keep",
                   help="how ties enter the agreement measures (default keep)")
    a.add_argument("--reps", type=int, default=1000, help="bootstrap resamples (default 1000)")
    a.add_argument("--seed", type=int, default=0, help="bootstrap seed (default 0)")
    a.add_argument("--json", action="store_true", help="print the full report as JSON")

    s = sub.add_parser("simulate", help="write judgments from a synthetic judge with known biases")
    s.add_argument("path", help="output JSONL file")
    s.add_argument("--pairs", type=int, default=200)
    s.add_argument("--position", type=float, default=0.0)
    s.add_argument("--length", type=float, default=0.0)
    s.add_argument("--self-pref", type=float, default=0.0)
    s.add_argument("--repeats", type=int, default=1)
    s.add_argument("--seed", type=int, default=0)

    r = sub.add_parser("run", help="judge pairs through an OpenAI-compatible chat endpoint")
    r.add_argument("path", help="JSONL of pairs: pair_id, prompt, answer_a, answer_b, ...")
    r.add_argument("--out", required=True, help="JSONL file for the judgments")
    r.add_argument("--base-url", default="http://localhost:8000/v1",
                   help="API root; the request goes to <base-url>/chat/completions")
    r.add_argument("--model", required=True, help="model name sent in the request")
    r.add_argument("--api-key-env", default="JUDGE_API_KEY",
                   help="environment variable holding the bearer token (default JUDGE_API_KEY)")
    r.add_argument("--judge-name", default=None,
                   help="judge name written into the judgments (default: the model name)")
    r.add_argument("--one-order", action="store_true",
                   help="show each pair only as AB instead of both orders")
    r.add_argument("--repeats", type=int, default=1, help="samples per pair and order")
    r.add_argument("--temperature", type=float, default=0.0)
    r.add_argument("--timeout", type=float, default=60.0, help="seconds per request")

    m = sub.add_parser("mtbench", help="convert the MT-Bench human-judgment parquet files")
    m.add_argument("--data-dir", required=True,
                   help="folder holding human.parquet and gpt4_pair.parquet (downloaded if absent)")
    m.add_argument("--humans", default=None, help="write judgments by the human judges here")
    m.add_argument("--judge", default=None,
                   help="write GPT-4 judgments paired with each human label here")
    m.add_argument("--turn", type=int, choices=(1, 2), default=None,
                   help="keep only verdicts on this turn (default: both turns)")
    m.add_argument("--human-pairs", default=None,
                   help="write human-against-human rows (an agreement baseline) here")
    m.add_argument("--inconsistent", choices=("tie", "drop"), default="tie",
                   help="order-inconsistent GPT-4 verdicts: keep as tie or drop (default tie)")
    return parser


def _cmd_run(args: argparse.Namespace) -> int:
    import os

    try:
        pairs = load_pairs(args.path)
    except (OSError, ValueError) as exc:
        print(f"judgebias: {exc}", file=sys.stderr)
        return 2
    judge = HttpJudge(
        base_url=args.base_url,
        model=args.model,
        api_key=os.environ.get(args.api_key_env) or None,
        temperature=args.temperature,
        timeout=args.timeout,
    )
    orders = ("AB",) if args.one_order else ("AB", "BA")
    try:
        judgments = run_pairs(
            pairs,
            judge,
            judge_name=args.judge_name or args.model,
            orders=orders,
            repeats=args.repeats,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"judgebias: {exc}", file=sys.stderr)
        return 1
    write_jsonl(args.out, to_rows(judgments))
    print(f"wrote {len(judgments)} judgments from {len(pairs)} pairs to {args.out}")
    return 0


def _cmd_mtbench(args: argparse.Namespace) -> int:
    from . import mtbench

    paths = mtbench.download(args.data_dir)
    judge_rows = mtbench.read_parquet(paths["gpt4_pair"])
    human_rows = mtbench.read_parquet(paths["human"])
    if args.turn is not None:
        judge_rows = [r for r in judge_rows if int(r["turn"]) == args.turn]
        human_rows = [r for r in human_rows if int(r["turn"]) == args.turn]
    counts = mtbench.summary(judge_rows)
    print(
        f"human rows {len(human_rows)}; judge rows {counts['rows']}: "
        f"decided {counts['decided']}, tie {counts['tie']}, "
        f"order-inconsistent {counts['inconsistent']} "
        f"({counts['inconsistent'] / counts['rows']:.3f} of pairs flipped under swap)"
    )
    if args.humans:
        rows = mtbench.convert_human(human_rows)
        write_jsonl(args.humans, rows)
        print(f"wrote {len(rows)} human judgments to {args.humans}")
    if args.human_pairs:
        rows = mtbench.convert_human_pairs(human_rows)
        write_jsonl(args.human_pairs, rows)
        print(f"wrote {len(rows)} human-human rows to {args.human_pairs}")
    if args.judge:
        rows = mtbench.convert_judge(judge_rows, human_rows, inconsistent=args.inconsistent)
        write_jsonl(args.judge, rows)
        print(f"wrote {len(rows)} judge judgments with human labels to {args.judge}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "simulate":
        rows = to_rows(
            simulate(
                args.pairs,
                position=args.position,
                length=args.length,
                self_pref=args.self_pref,
                repeats=args.repeats,
                seed=args.seed,
            )
        )
        write_jsonl(args.path, rows)
        print(f"wrote {len(rows)} judgments to {args.path}")
        return 0
    if args.command == "run":
        return _cmd_run(args)
    if args.command == "mtbench":
        return _cmd_mtbench(args)
    try:
        judgments = load_jsonl(args.path)
    except (OSError, ValueError) as exc:
        print(f"judgebias: {exc}", file=sys.stderr)
        return 2
    if not judgments:
        print("judgebias: no judgments found", file=sys.stderr)
        return 2
    report = audit(judgments, ties=args.ties, reps=args.reps, seed=args.seed)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(render(report))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
