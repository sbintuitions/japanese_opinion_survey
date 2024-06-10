from scipy.spatial import distance
from src.analysis.items import Result
import numpy as np
from scipy.stats import entropy
import math


def calculate_average_maximum_probability(results: list[Result]) -> float:
    probabilities = [max([math.exp(log_prob) for log_prob in result.log_probs]) for result in results]
    return np.mean(probabilities)

def calculate_average_distributions_per_label(results: list[Result]) -> float:
    return list(np.mean([result.distributions for result in results], axis=0))

def calculate_averaged_js_divergence(results_with_same_id: list[Result]):
    js_divergences = []
    for i, result1 in enumerate(results_with_same_id):
        for j, result2 in enumerate(results_with_same_id[i + 1:]):
            if np.linalg.norm(np.array(result1.distributions) - np.array(result2.distributions)) < 10**(-8):
                js_divergence = 0.0
            else:
                try:
                    js_divergence = distance.jensenshannon(result1.distributions, result2.distributions)
                except Exception as e:
                    print(e)
            js_divergences.append(js_divergence)
    return np.mean(js_divergences)


def calculate_averaged_consistency_with_same_id(results_with_same_id: list[Result]):
    js_divergences = []
    consistencies = []
    for i, result1 in enumerate(results_with_same_id):
        for j, result2 in enumerate(results_with_same_id[i + 1:]):
            if np.linalg.norm(np.array(result1.distributions) - np.array(result2.distributions)) < 10**(-8):
                js_divergence = 0.0
            else:
                try:
                    js_divergence = distance.jensenshannon(result1.distributions, result2.distributions)
                except Exception as e:
                    print(e)
            js_divergences.append(js_divergence)
            if np.argmax(np.array(result1.distributions)) == np.argmax(np.array(result2.distributions)):
                consistencies.append(1)
            else:
                consistencies.append(0)
    return np.mean(js_divergences), np.mean(consistencies)


def calculate_average_maximum_probability_with_same_id(results_with_same_id: list[Result]) -> float:
    return max([math.exp(log_prob) for log_prob in results_with_same_id[0].log_probs])


def calculate_average_sum_probability_with_same_id(results_with_same_id: list[Result]) -> float:
    return sum([math.exp(log_prob) for log_prob in results_with_same_id[0].log_probs])


def calculate_average_distributions(results_with_same_id: list[Result]):
    return np.mean([result.distributions for result in results_with_same_id], axis=0)


def calculate_average_distributions_per_label_with_same_id(results_with_same_id: list[Result]):
    return np.mean([result.distributions_per_label for result in results_with_same_id], axis=0)


def calculate_average_entropy(results_with_same_id: list[Result]):
    return entropy(np.mean([result.distributions for result in results_with_same_id], axis=0))


def calculate_average_entropy_per_label(results_with_same_id: list[Result]):
    return entropy(np.mean([result.distributions_per_label for result in results_with_same_id], axis=0))


def calculate_distributions_from_log_probs(result: Result, by_choices: bool):
    if by_choices:
        probabilities = [math.exp(log_prob) for log_prob in result.log_probs]
        probabilities_normalized = [probability / sum(probabilities) for probability in probabilities]
        distributions = probabilities_normalized
        distributions_per_label = [0 for _ in range(len(result.permutation))]
        for order, probability in zip(result.permutation, probabilities_normalized):
            distributions_per_label[order] = probability
    else:
        probabilities = [math.exp(log_prob) for log_prob in result.log_probs]
        probabilities_normalized = [probability / sum(probabilities) for probability in probabilities]
        distributions_per_label = probabilities_normalized
        distributions = [0 for _ in range(len(result.permutation))]
        for order, probability in zip(result.permutation, probabilities_normalized):
            distributions[order] = probability
    return distributions_per_label, distributions
