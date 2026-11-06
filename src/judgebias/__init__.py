"""judgebias: audit an LLM judge before trusting it."""

from .data import Judgment, judgment_from_dict, load_jsonl, write_jsonl
from .measures import (
    agreement,
    audit,
    distinct,
    length_bias,
    position_bias,
    repeat_consistency,
    self_preference,
)
from .stats import Estimate, bootstrap, cohen_kappa, logistic_fit, logistic_fit_multi

__version__ = "0.1.0"

__all__ = [
    "Estimate",
    "Judgment",
    "__version__",
    "agreement",
    "audit",
    "bootstrap",
    "cohen_kappa",
    "distinct",
    "judgment_from_dict",
    "length_bias",
    "load_jsonl",
    "logistic_fit",
    "logistic_fit_multi",
    "position_bias",
    "repeat_consistency",
    "self_preference",
    "write_jsonl",
]
