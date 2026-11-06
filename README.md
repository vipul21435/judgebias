# judgebias

Audit an LLM judge for position bias, length bias, self-preference and agreement with human labels before trusting it.

Status: first version. The measures and the `audit` command work on JSONL files of pairwise judgments and on a synthetic judge with known biases. Results on public human-judgment data are not in yet; the next version adds them.

## Why this is hard

A pairwise judge is scored on a sample of prompts, each shown in one order, so a judge that prefers whichever answer comes first looks like it has an opinion about the answers. The same holds for length: longer answers win more often with most judges, and without the lengths in hand that reads as quality. A judge that is itself a model also tends to favour answers from its own family. Human labels help, but they disagree with each other too and contain ties, which most agreement numbers either drop silently or count as errors. None of these effects is visible from a single accuracy figure, and small samples make a point estimate without an interval close to meaningless.

## Install

```
pip install git+https://github.com/vipul21435/judgebias
```

Python 3.10 or newer; depends on numpy and scipy.

## Quickstart

Each line of the input is one judgment. `verdict` is `A`, `B` or `tie` and names the answer, not the slot it was shown in; `order` is `AB` (A shown first, the default) or `BA`. `len_a`, `len_b`, `model_a`, `model_b`, `human` and `repeat` are optional and switch on the measures that need them. A few common key spellings are accepted (`winner` for `verdict`, `question_id` for `pair_id`, `answer_a` text in place of `len_a`).

```
{"pair_id": "p00000", "verdict": "A", "order": "AB", "judge": "gamma-70b", "len_a": 312, "len_b": 190, "model_a": "alpha-7b", "model_b": "beta-13b", "human": "A", "repeat": 0}
```

`judgebias simulate` writes judgments from a synthetic judge with chosen biases; `judgebias audit` measures them. The example file in `examples/` was made with `--pairs 60 --position 0.6 --length 1.0 --seed 11`:

```
$ judgebias audit examples/synthetic_judge.jsonl
judgments: 120  judges: gamma-70b
position bias
  first-position win rate  0.647 [0.592, 0.706] n=60  (decided 119, ties 1)
  flip rate under swap     0.300 [0.183, 0.417] n=60  (pairs with both orders 60)
  first-both among flips   0.944 [0.812, 1.000] n=60
length bias
  longer-answer win rate   0.748 [0.664, 0.831] n=60
  logistic slope (log len ratio)  1.739 [1.128, 2.670] n=60
self-preference (judge family gamma)
  own-family win rate      0.540 [0.393, 0.682] n=60
  same, by human labels    0.467 [0.273, 0.643] n=60
  excess                   +0.073
agreement with humans (ties=keep, labelled 120)
  raw agreement            0.683 [0.583, 0.783] n=60
  Cohen's kappa            0.410 [0.237, 0.581] n=60
```

Brackets are 95 % bootstrap intervals; `n` is the number of prompts resampled. `--json` prints the full report, `--ties drop` removes tied rows from the agreement measures, `--reps` and `--seed` control the bootstrap.

From Python (`examples/run_audit.py`):

```
$ uv run python examples/run_audit.py
first-position win rate: 0.723 [0.696, 0.753]
length slope: 1.909 [1.644, 2.206]
kappa vs human labels: 0.317 [0.244, 0.385]
fair judge first-position win rate: 0.500 [0.500, 0.500]
```

The last line is exact rather than lucky: a judge seen in both orders that never flips picks the first slot exactly half the time, so the interval collapses.

## How it works

Every measure resamples prompts, not judgments, so both presentation orders and repeated samples of one pair stay together in each bootstrap draw. Intervals are percentile intervals from 1000 draws by default.

- Position bias. `first_win_rate` is the share of decided verdicts that went to the answer shown first; 0.5 is unbiased. `flip_rate` is, over pairs judged in both orders, the share whose verdict changed with the order. `first_both_rate` is, among flipped pairs, the share where the judge took the first slot both times rather than the second.
- Length bias. `longer_win_rate` is the share of decided verdicts with unequal lengths that went to the longer answer. `slope` is the coefficient b in a logistic fit P(A wins) = sigmoid(a + b x) with x = log((len_a + 1) / (len_b + 1)), fitted by L-BFGS on the penalised log-likelihood (L2 weight 1e-4 on b, so separable data gives a finite number). b = 0 means length does not move the verdict.
- Self-preference. A model's family is its name up to the first `-`, `/`, `_` or `:`. On judgments where exactly one candidate shares the judge's family, `own_win_rate` is how often the judge picked that candidate, and `own_win_rate_human` is how often the human label did; `excess` is the difference. A positive excess means the judge favours its own family beyond what humans see.
- Agreement. Raw agreement and Cohen's kappa between judge and human labels. With `ties=keep`, tie is a third category on both sides; with `ties=drop`, any row where either side said tie is removed. Kappa is (po - pe) / (1 - pe) over the full confusion table.
- Repeat consistency. For (pair, order) groups with two or more samples, the share of unanimous groups and the share of agreeing sample pairs.

## Results on real public data

Not yet. The planned data is a public human-judgment set of pairwise model comparisons; the numbers, the exact command and the licence will go here once the run has been made.

## Limitations

- Position and length bias are measured on the judge's verdicts alone; they do not separate a judge's preference from a real correlation between position or length and quality in the data. The swap-based measures need both orders present.
- The logistic fit has one covariate. Confounders such as the model identities are not adjusted for.
- Self-preference uses a name-prefix family rule, which is wrong for models whose names do not start with their vendor or family.
- The bootstrap is a percentile interval; for very small samples or rates near 0 or 1 it under-covers.
- No judge is called over the network in this version; the run mode with an HTTP judge is not implemented.

## Related tools

- https://github.com/vipul21435/evalstats
- https://github.com/vipul21435/prefpairs
- https://github.com/vipul21435/rubricops
- https://github.com/vipul21435/graderguard

## Licence

MIT.

Built by Vipul Raj Jha · https://vipul21435.github.io
