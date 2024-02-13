import argparse
from collections import defaultdict
from dataclasses import asdict

import jsonlines
import numpy as np
import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer, pipeline,
                          set_seed)

from src.distributions import get_model_distributions
from src.load_data import load_japanese_opinion_survey
from src.prompt_creators import create_japanese_opinion_survey_prompt
from src.utils import make_permutations
from src.generations import generate_model_distributions_gpt

parser = argparse.ArgumentParser()

parser.add_argument("--model", type=str)
parser.add_argument("--dataset", type=str, default="japanese_opinion_survey")
parser.add_argument("--temperature", type=float, default=0.7)
parser.add_argument("--opinion_type", type=str, default="distribution", choices=["distribution", "generation"])
parser.add_argument("--label_type", type=str, default="number")
parser.add_argument("--include_reject", type=bool, default=False)
parser.add_argument("--include_succession", type=bool, default=False)
parser.add_argument("--output_path", type=str)

args = parser.parse_args()

if "gpt" in args.model:
    model = args.model
else:
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, device_map="auto")
    pipe = pipeline("text-generation", model=args.model_path, torch_dtype=torch.bfloat16, device_map="auto")

if args.dataset == "japanese_opinion_survey":
    qas = load_japanese_opinion_survey("resources/yorontyousa_trimmed.csv", include_reject=args.include_reject, include_succession=args.include_succession)
else:
    qas = []

if args.label_type == "number":
    labels = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]
else:
    labels = ["ア", "イ", "ウ", "エ", "オ", "カ", "キ", "ク", "ケ", "コ", "サ", "シ", "ス", "セ", "ソ", "タ", "チ", "ツ", "テ", "ト", "ナ", "ニ", "ヌ", "ネ", "ノ", "ハ", "ヒ", "フ", "ヘ", "ホ", "マ", "ミ", "ム", "メ", "モ", "ヤ", "ユ", "ヨ"]

output_qas = []
for qa in qas:
    permutations = make_permutations(qa=qa)
    distribution_dict = defaultdict(list)
    for permutation in permutations:
        prompt = create_japanese_opinion_survey_prompt(qa, labels, permutation=permutation)
        if args.opinion_type == "generation":
            distribution = generate_model_distributions_gpt(prompt=prompt, model=model, params={"temperature": args.temperature}, labels=labels)
        else:
            distribution = get_model_distributions(prompt=prompt, model=model, tokenizer=tokenizer, params={}, labels=labels)
        for i in range(len(permutation)):
            distribution_dict[qa.choices[permutation[i]].content].append(float(distribution[i]))
    for i in range(len(qa.choices)):
        qa.choices[i].predicted_distributions = distribution_dict[qa.choices[i].content]
    output_qas.append(qa)

with jsonlines.open(args.output_path, "w") as writer:
    for qa in output_qas:
        writer.write(asdict(qa))
