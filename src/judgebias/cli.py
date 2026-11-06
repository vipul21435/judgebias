"""Command line entry point: ``judgebias audit`` and ``judgebias simulate``."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Sequence
from typing import Any

from . import __version__
from .data import load_jsonl, write_jsonl
from .measures import audit
from .synth import simulate, to_rows


def _fmt(est: dict[str, Any]) -> str:
    v = est["value"]
    if isinstance(v, float) and math.isnan(v):
        return "n/a"
    return f"{v:.3f} [{est['low']:.3f}, {est['high']:.3f}] n={est['n']}"


def render(report: dict[str, Any]) -> str:
    lines = [f"judgments: {report['n_judgments']}  judges: {', '.join(report['judges'])}"]
    p = report["position"]
    lines.append("position bias")
    lines.append(
        f"  first-position win rate  {_fmt(p['first_win_rate'])}"
        f"  (decided {p['n_decided']}, ties {p['n_ties']})"
    )
    lines.append(
        f"  flip rate under swap     {_fmt(p['flip_rate'])}"
        f"  (pairs with both orders {p['n_pairs_both_orders']})"
    )
    lines.append(f"  first-both among flips   {_fmt(p['first_both_rate'])}")
    if "length" in report:
        ln = report["length"]
        lines.append("length bias")
        lines.append(f"  longer-answer win rate   {_fmt(ln['longer_win_rate'])}")
        lines.append(f"  logistic slope (log len ratio)  {_fmt(ln['slope'])}")
    if "self_preference" in report:
        s = report["self_preference"]
        lines.append(f"self-preference (judge family {s['judge_family']})")
        lines.append(f"  own-family win rate      {_fmt(s['own_win_rate'])}")
        if "own_win_rate_human" in s:
            lines.append(f"  same, by human labels    {_fmt(s['own_win_rate_human'])}")
            lines.append(f"  excess                   {s['excess']:+.3f}")
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
    return parser


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
