# Changelog

## 0.1.0 - 2026-11-06

### Added

- Pairwise judgment loader for JSONL files, with common key spellings and errors that name the file and line. Numeric verdict labels and the slot words `first` / `second` are refused.
- Position bias: first-position win rate, flip rate under swap (compared only within the same judge and repeat), first-both and second-both shares among flips, and the share of order-inconsistent verdicts that already combine both orders (`flipped`).
- Length bias: longer-answer win rate and a logistic slope on the log length ratio, raw and adjusted for the human label.
- Self-preference by model family, with the excess over human labels; `--family NAME=FAMILY` sets a family by hand and vendor prefixes such as `openai/` are ignored.
- Agreement with human labels: raw agreement and Cohen's kappa, with ties kept or dropped.
- Repeat consistency per pair, order and judge.
- Bootstrap percentile intervals that resample prompts, so both orders, repeats and several human labels of one pair stay together; draws with an undefined statistic are left out of the interval.
- Audit notes and warnings for one-order files, rows that share a key but differ in verdict, and unparsed replies; `audit --judge NAME` keeps one judge's rows; `audit --json` writes strict JSON.
- `judgebias audit`, `judgebias simulate` (synthetic judge with known biases), `judgebias run` (OpenAI-compatible chat endpoint, both orders, with a fixed-pattern reply parser, `--unparsed`, `--max-tokens`, `--system-file` and `--template-file`) and `judgebias mtbench` (converter for the MT-Bench human-judgment data, with a human-human baseline).
- Results on the public MT-Bench human judgments in the README, with commands, bootstrap intervals and a check of the agreement figures against Table 5 of the MT-Bench paper.
- Examples: Python audit, length-bias chart (`docs/length_bias.png`), local stub judge server, and a simulation of interval coverage (`examples/bootstrap_coverage.py`).
- An offline test suite with fake transports, synthetic judges, hand-built MT-Bench rows and malformed input files.

### Known limitations

See the Limitations section of the README; among other things, the run mode has been tried only against the local stub server.
