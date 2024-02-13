from src.items import QA


def create_japanese_opinion_survey_prompt(qa: QA, labels: list[str], permutation: list[int]) -> str:
    prompt = f"""{qa.prefix}

{qa.question}

"""
    for i in range(len(qa.choices)):
        prompt += f"{labels[i]} {qa.choices[permutation[i]].content}\n"
    prompt += "\n"
    prompt += "回答: "
    return prompt
