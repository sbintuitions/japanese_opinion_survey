from dataclasses import dataclass
from typing import Optional, List


@dataclass
class Choice:
    content: str
    real_distribution: float
    predicted_distributions: List[float]


@dataclass
class QA:
    prefix: str
    question: str
    choices: list[Choice]
    year: str
    topic: str
    question_class: str
