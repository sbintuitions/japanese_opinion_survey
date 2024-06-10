from dataclasses import dataclass
from typing import Optional

@dataclass
class Result:
    id: int
    question: str
    prefix: str
    choices: list[str]
    labels: list[str]
    permutation: list[int]
    log_probs: list[float]
    byte_norm_log_probs: list[float]
    distributions: Optional[list[float]] = None
    distributions_per_label: Optional[list[float]] = None
