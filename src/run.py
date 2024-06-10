from pathlib import Path
from flexeval.core.multiple_choice_dataset.base import MultipleChoiceDataset
from tqdm import tqdm
from items import MyChoiceDataset, QA
from flexeval.core.language_model.base import LanguageModel
from flexeval.scripts.common import (
    save_json,
)
from jsonargparse import ActionConfigFile, ArgumentParser
import logging
from load_data import load_japanese_opinion_survey
from prompt_creators import create_japanese_opinion_survey_prompt
from utils import make_permutations
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(filename)s:%(lineno)d %(message)s",
)
logger = logging.getLogger(__name__)

NUM_LABELS = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20"]
ALPHABET_LABELS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T"]
KATAKANA_LABELS = ["ア", "イ", "ウ", "エ", "オ", "カ", "キ", "ク", "ケ", "コ", "サ", "シ", "ス", "セ", "ソ", "タ", "チ", "ツ", "テ", "ト", "ナ", "ニ", "ヌ", "ネ", "ノ", "ハ", "ヒ", "フ", "ヘ", "ホ", "マ", "ミ", "ム", "メ", "モ", "ヤ", "ユ", "ヨ"]
HIRAGANA_LABELS = ["あ", "い", "う", "え", "お", "か", "き", "く", "け", "こ"]

def predict_multiple_choice(
    language_model: LanguageModel,
    eval_dataset: MultipleChoiceDataset,
    labels: list[str] | None = None,
    save_dir: str | None = None,
    all_permutations: bool = False,
    measure_by_choices: bool = False,
) -> dict[str, float]:
    results: list[dict[str, Any]] = []
    with tqdm(total=len(eval_dataset)) as pbar:
        for i, instance in enumerate(eval_dataset):
            permutations = make_permutations(instance, all_permutations=all_permutations)
            for permutation in permutations:
                batch_prefixes = [create_japanese_opinion_survey_prompt(instance.inputs["question"], choices=instance.choices, labels=labels, permutation=permutation)] * len(instance.choices)
                if measure_by_choices:
                    text_list = instance.choices
                else:
                    text_list = labels[:len(instance.choices)]
                log_probs_for_choices = language_model.batch_compute_log_probs(
                    text_list=text_list, prefix_list=batch_prefixes
                )
                # select the choice with the highest log probability as model output
                max_log_prob = max(log_probs_for_choices)
                max_log_prob_index = log_probs_for_choices.index(max_log_prob)

                # we also calculate accuracy using byte-normalized log probabilities
                # for the discussion on normalization methods, see
                # https://github.com/EleutherAI/lm-evaluation-harness/issues/1396
                # https://blog.eleuther.ai/multiple-choice-normalization/
                norm_log_probs = [
                    log_p / len(choice.encode("utf-8"))
                    for log_p, choice in zip(log_probs_for_choices, text_list)
                ]
                max_norm_log_p = max(norm_log_probs)
                max_norm_log_p_index = norm_log_probs.index(max_norm_log_p)

                results.append(
                    {
                        "ID": i,
                        "question": instance.inputs["question"],
                        "prefix": batch_prefixes[0],
                        "choices": instance.choices,
                        "labels": text_list,
                        "permutation": permutation,
                        "log_probs": log_probs_for_choices,
                        "prediction": max_log_prob_index,
                        "byte_norm_log_probs": norm_log_probs,
                        "byte_norm_prediction": max_norm_log_p_index,
                        "log_prob_sum": sum(log_probs_for_choices),
                        "measure_by_choices": measure_by_choices
                    }
                )
            if i % 100 == 0:
                print(results[-1])
            if i % 10 == 0:
                pbar.update(i)

    return results


def evaluate(language_model: LanguageModel, qas: list[QA], labels: list[str], all_permutations: bool = False, measure_by_choices: bool = False):
    dataset = MyChoiceDataset(qas)
    
    results = predict_multiple_choice(language_model, dataset, labels=labels, all_permutations=all_permutations, measure_by_choices=measure_by_choices)
    return {
        "results": results
    }


if __name__ == "__main__":
    parser = ArgumentParser(parser_mode="jsonnet")

    parser.add_subclass_arguments(LanguageModel, nested_key="language_model", required=True)
    parser.add_argument("--input", type=str, default="./resources/v3.csv")
    parser.add_argument("--save_dir", type=str, default=None, help="Directory to save the outputs")
    parser.add_argument("--force", type=bool, default=False, help="Overwrite the save_dir if it exists")
    parser.add_argument("--config", action=ActionConfigFile)
    parser.add_argument('--all_permutations', action='store_true', help="If true, we calculate all permutations of the choices")
    parser.add_argument('--measure_by_choices', action='store_true', help="If true, we directly calculate the probability of the choice contents instead of choice labels")
    parser.add_argument("--labels_type", type=str, default="NUMBER")
    
    args = parser.parse_args()
    logger.info(args)

    args_as_dict = args.as_dict()
    args = parser.instantiate_classes(args)

    if args.labels_type == "ALPHABET":
        labels = ALPHABET_LABELS
    elif args.labels_type == "KATAKANA":
        labels = KATAKANA_LABELS
    elif args.labels_type == "HIRAGANA":
        labels = HIRAGANA_LABELS
    else:
        labels = NUM_LABELS

    qas = load_japanese_opinion_survey(file_path=args.input, include_reject=False, include_succession=False)

    results = evaluate(
        args.language_model, qas=qas, labels=labels, all_permutations=args.all_permutations, measure_by_choices=args.measure_by_choices
    )
    
    if args.save_dir is not None:
        save_json(args_as_dict, Path(args.save_dir) / "config.json")
        save_json(results, Path(args.save_dir) / "results.json")

    logger.info("done")
