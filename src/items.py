from dataclasses import dataclass
from flexeval.core.multiple_choice_dataset.base import MultipleChoiceDataset, MultipleChoiceInstance


@dataclass
class Choice:
    content: str
    real_distribution: float
    predicted_distributions: list[float]


@dataclass
class QA:
    prefix: str
    question: str
    choices: list[Choice]
    year: str
    topic: str
    question_class: str
    question_type: str


class MyChoiceDataset(MultipleChoiceDataset):
    def __init__(
        self,
        qas
    ):
        self._qas = qas

    def __len__(self) -> int:
        return len(self._qas)

    def __getitem__(self, i: int) -> MultipleChoiceInstance:
        question = self._qas[i]
        
        return MultipleChoiceInstance(
            inputs={"question": question.question},
            choices=[choice.content for choice in question.choices],
            correct_choice_index=0,
        )
