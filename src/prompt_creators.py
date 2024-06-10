def create_japanese_opinion_survey_prompt(question: str, choices: list[str], labels: list[str], permutation: list[int]) -> str:
    prompt = f"""質問: {question}
"""
    for i in range(len(choices)):
        prompt += f"{labels[i]}: {choices[permutation[i]]}\n"
    prompt += "回答: "
    return prompt
