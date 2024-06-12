from ast import literal_eval
from datasets import load_dataset

import pandas as pd

from src.items import QA, Choice


def load_japanese_opinion_survey(file_path: str, include_reject: bool, include_succession: bool) -> list[QA]:
    df = pd.read_csv(file_path)
    qas = []
    if include_reject:
        reject_choices = []
    else:
        reject_choices = ["わからない", "無回答"]
    for heading, context_information, parent_question, question_number, question, choices, distributions, year, topic, question_class, question_type in zip(df["Heading"], df["Context Information"], df["Parent Question"], df["Question Number"], df["Question"], df["Choices"], df["Distributions"], df["Year"], df["Topic"], df["Question Class"], df["Question Type"]):
        if not include_succession and isinstance(parent_question, str):
            continue
        if question_type != "SA":
            continue
        filtered_choices = []
        for content, distribution in zip(literal_eval(choices), literal_eval(distributions)):
            if content in reject_choices:
                continue
            filtered_choices.append(Choice(content=content, real_distribution=distribution, predicted_distributions=0.0))
        qas.append(QA(heading=heading, context_information=context_information, question=question, choices=filtered_choices, year=year, topic=topic, question_class=question_class, question_type=question_type, question_number=question_number))
    return qas


def load_japanese_opinion_survey_hf(dataset_name: str, include_reject: bool, include_succession: bool) -> list[QA]:
    datasets = load_dataset(dataset_name)["test"]
    qas = []
    if include_reject:
        reject_choices = []
    else:
        reject_choices = ["わからない", "無回答"]
    for dataset in datasets:
        if not include_succession and isinstance(dataset["Parent Question"], str):
            continue
        if dataset["Question Type"] != "SA":
            continue
        filtered_choices = []
        for choice, distribution in zip(dataset["Choices"], dataset["Distributions"]):
            if choice in reject_choices:
                continue
            filtered_choices.append(Choice(content=choice, real_distribution=distribution, predicted_distributions=0.0))
        qas.append(QA(context_information=dataset["Context Information"], heading=dataset["Heading"], question=dataset["Question"], choices=filtered_choices, year=dataset["Year"], topic=dataset["Topic"], question_class=dataset["Question Class"], question_type=dataset["Question Type"], question_number=dataset["Question Number"]))
    return qas
