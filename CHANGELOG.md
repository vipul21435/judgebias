# Changelog

## 0.1.0

- Pairwise judgment loader (JSONL) with common key spellings.
- Position bias: first-position win rate, flip rate under swap, first-both share among
  flips, and the flip rate of verdicts that already combine both orders (`flipped`).
- Length bias: longer-answer win rate and a logistic slope on the log length ratio, raw and
  adjusted for the human label, with bootstrap intervals over prompts.
- Self-preference by model family, with the excess over human labels and its interval.
- Agreement with human labels: raw and Cohen's kappa, ties kept or dropped, bootstrap
  intervals over prompts.
- Repeat consistency per pair, order and judge.
- `judgebias audit`, `judgebias simulate` (synthetic judge with known biases),
  `judgebias run` (OpenAI-compatible chat endpoint, both orders) and `judgebias mtbench`
  (converter for the MT-Bench human-judgment data, with a human-human baseline).
- Examples: Python audit, length-bias chart, local stub judge server.
