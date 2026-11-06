"""Audit a synthetic judge whose biases are known, from Python.

Run from the repository root:

    uv run python examples/run_audit.py
"""

from judgebias import audit, position_bias
from judgebias.synth import simulate

# A judge that leans toward the first-shown answer and toward longer answers.
judgments = simulate(300, position=0.8, length=1.5, seed=0)

report = audit(judgments, reps=500)
pos = report["position"]["first_win_rate"]
print(f"first-position win rate: {pos['value']:.3f} [{pos['low']:.3f}, {pos['high']:.3f}]")
slope = report["length"]["slope"]
print(f"length slope: {slope['value']:.3f} [{slope['low']:.3f}, {slope['high']:.3f}]")
kappa = report["agreement"]["kappa"]
print(f"kappa vs human labels: {kappa['value']:.3f} [{kappa['low']:.3f}, {kappa['high']:.3f}]")

# The same judge with the biases switched off, for comparison.
fair = position_bias(simulate(300, seed=0), reps=500)["first_win_rate"]
print(
    f"fair judge first-position win rate: {fair['value']:.3f} "
    f"[{fair['low']:.3f}, {fair['high']:.3f}]"
)
