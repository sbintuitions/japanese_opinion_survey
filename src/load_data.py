from ast import literal_eval
from typing import List

import pandas as pd

from src.items import QA, Choice


def load_japanese_opinion_survey(file_path: str, include_reject: bool, include_succession: bool) -> List[QA]:
    df = pd.read_csv(file_path)
    qas = []
    if include_reject:
        reject_choices = []
    else:
        reject_choices = ["わからない", "無回答"]
    for prefix, is_succession, question, choices, distributions, year, topic, question_class in zip(df["prefix"], df["is_succession"], df["question"], df["choices"], df["distributions"], df["year"], df["topic"], df["question_class"]):
        if not include_succession and is_succession == "YES":
            continue
        contents = literal_eval(choices)
        float_distributions = literal_eval(distributions)
        choices = []
        for content, distribution in zip(contents, float_distributions):
            if content in reject_choices:
                continue
            choices.append(Choice(content=content, real_distribution=distribution, predicted_distributions=[]))
        qas.append(QA(prefix=prefix, question=question, choices=choices, year=year, topic=topic, question_class=question_class))
    return qas
