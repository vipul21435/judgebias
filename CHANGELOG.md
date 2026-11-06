# Changelog

## Unreleased

- Flip rate and first-both share compare an AB sample only with the BA sample of the same
  judge and repeat; files with several judges no longer show flips between judges.
  `judgebias audit --judge NAME` keeps one judge's rows.
- Run mode: the reply parser reads a bare `A` / `B` only when it is the whole reply, accepts
  `[[A]]` / `[[B]]` / `[[C]]` markers, and counts replies that name both slots or none as
  unparsed instead of as ties (`--unparsed PATH` writes them out). Judgments are written to
  `--out` as they arrive, so a failure partway keeps the earlier ones.
- Numeric verdict labels (`0`, `1`, `2`) are refused with an error.
- `judgebias mtbench` prints download and parquet errors as one line.
- Logistic slope is nan, not 0, on a resample with one verdict class, so such resamples
  drop out of the interval; a fit that does not converge raises a warning.
- `examples/bootstrap_coverage.py` and a test check interval coverage by simulation.

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
