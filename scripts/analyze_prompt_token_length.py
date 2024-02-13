import argparse
from collections import defaultdict
from dataclasses import asdict

import jsonlines
import numpy as np
import tiktoken

from src.distributions import get_model_distributions
from src.load_data import load_japanese_opinion_survey
from src.prompt_creators import create_japanese_opinion_survey_prompt
from src.utils import make_permutations

parser = argparse.ArgumentParser()


qas = load_japanese_opinion_survey("resources/yorontyousa_trimmed.csv")

labels = ["ア", "イ", "ウ", "エ", "オ", "カ", "キ", "ク", "ケ", "コ", "サ", "シ", "ス", "セ", "ソ", "タ", "チ", "ツ", "テ", "ト", "ナ", "ニ", "ヌ", "ネ", "ノ", "ハ", "ヒ", "フ", "ヘ", "ホ", "マ", "ミ", "ム", "メ", "モ", "ヤ", "ユ", "ヨ"]
enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
output_lengths = []
for qa in qas:
    permutations = make_permutations(qa=qa)
    distribution_dict = defaultdict(list)
    for permutation in permutations:
        prompt = create_japanese_opinion_survey_prompt(qa, labels, permutation=permutation)
        tokens = enc.encode(prompt)
        output_lengths.append(len(tokens))
print(sum(output_lengths) / len(output_lengths))
print(sum(output_lengths) / 1000 * 0.0005)
input_cost = sum(output_lengths) / 1000 * 0.0005
output_cost = len(output_lengths) / 1000 * 20 * 0.0015
print(input_cost + output_cost)
