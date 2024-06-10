import argparse
import os
import numpy as np
from src.analysis.load_data import load_results
from src.analysis.utils import calculate_average_maximum_probability, calculate_average_entropy, calculate_average_entropy_per_label, calculate_average_distributions_per_label, calculate_average_maximum_probability_with_same_id, calculate_average_sum_probability_with_same_id, calculate_averaged_consistency_with_same_id
from src.analysis.write_data import write_analyses_csv, write_analyses_json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--experiment_setting", type=str)
    parser.add_argument("--by_choices", action='store_true')
    args = parser.parse_args()
    result_dir = os.path.join("results/request", args.experiment_setting)
    institutes = os.listdir(result_dir)
    analyses = {}
    for institute in institutes:
        if institute == ".DS_Store":
            continue
        institute_dir = os.path.join(result_dir, institute)
        models = list(sorted(os.listdir(institute_dir)))
        for model in models:
            if model == ".DS_Store":
                continue
            analyse = {}
            model_dir = os.path.join(institute_dir, model)
            results = load_results(os.path.join(model_dir, "results.json"), args.by_choices)
            analyse["average_maximum_probability"] = calculate_average_maximum_probability(results=results)
            analyse["average_distributions_per_label"] = calculate_average_distributions_per_label(results=results)
            ids = sorted(list(set([result.id for result in results])))
            results_by_id = [[result for result in results if result.id == id] for id in ids]
            js_divergences = []
            consistencies = []
            entropy_by_choices = []
            entropy_by_labels = []
            average_maximum_probabilities_with_same_id = []
            average_sum_probabilities_with_same_id = []
            for results_with_same_id in results_by_id:
                js_divergence, consistency = calculate_averaged_consistency_with_same_id(results_with_same_id=results_with_same_id)
                js_divergences.append(js_divergence)
                consistencies.append(consistency)
                entropy_by_choices.append(calculate_average_entropy(results_with_same_id=results_with_same_id))
                entropy_by_labels.append(calculate_average_entropy_per_label(results_with_same_id=results_with_same_id))
                average_maximum_probabilities_with_same_id.append(calculate_average_maximum_probability_with_same_id(results_with_same_id=results_with_same_id))
                average_sum_probabilities_with_same_id.append(calculate_average_sum_probability_with_same_id(results_with_same_id=results_with_same_id))
            analyse["averaged_js_divergences"] = np.mean(js_divergences)
            analyse["averaged_consistencies"] = np.mean(consistencies)
            analyse["averaged_entropy_by_choices"] = np.mean(entropy_by_choices)
            analyse["averaged_entropy_by_labels"] = np.mean(entropy_by_labels)
            analyse["average_maximum_probability_with_same_id"] = np.mean(average_maximum_probabilities_with_same_id)
            analyse["average_sum_probabilities_with_same_id"] = np.mean(average_sum_probabilities_with_same_id)
            analyses[model] = analyse
    analysis_dir = os.path.join("results/analysis", args.experiment_setting)
    if not os.path.exists(analysis_dir):
        os.mkdir(analysis_dir)
    write_analyses_json(analyses=analyses, file_path=os.path.join(analysis_dir, "results.json"))
    write_analyses_csv(analyses=analyses, file_path=os.path.join(analysis_dir, "results.csv"))
