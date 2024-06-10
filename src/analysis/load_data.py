import json
from src.analysis.items import Result
from src.analysis.utils import calculate_distributions_from_log_probs

def load_results(file_path, by_choices: bool=False) -> list[Result]:
    with open(file_path) as f:
        result_dicts = json.load(f)["results"]
    results = []
    for result_dict in result_dicts:
        result = Result(id=result_dict["ID"], 
                        question=result_dict["question"], 
                        prefix=result_dict["prefix"], 
                        choices=result_dict["choices"],
                        labels=result_dict["labels"],
                        permutation=result_dict["permutation"],
                        log_probs=result_dict["log_probs"],
                        byte_norm_log_probs=result_dict["byte_norm_log_probs"])
        distributions_per_label, distributions = calculate_distributions_from_log_probs(result=result, by_choices=by_choices)
        result.distributions_per_label = distributions_per_label
        result.distributions = distributions
        results.append(result)
    return results
