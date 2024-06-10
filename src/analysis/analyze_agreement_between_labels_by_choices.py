from src.analysis.load_data import load_results
import os
from scipy.spatial import distance
import numpy as np
from src.analysis.write_data import write_analyses_csv, write_analyses_json
from collections import defaultdict


jcqa_number_result_dir = "results/request/jcqa_all_permutations_numbers_by_choices"
jcqa_katakana_result_dir = "results/request/jcqa_all_permutations_katakana_by_choices"

v3_number_result_dir = "results/request/v3_all_permutations_numbers_by_choices"
v3_katakana_result_dir = "results/request/v3_all_permutations_katakana_by_choices"



institutes = os.listdir(jcqa_number_result_dir)
analyses = {}
for institute in institutes:
    jcqa_number_institute_dir = os.path.join(jcqa_number_result_dir, institute)
    jcqa_katakana_institute_dir = os.path.join(jcqa_katakana_result_dir, institute)
    models = list(sorted(os.listdir(jcqa_number_institute_dir)))
    for model in models:
        jcqa_number_model_dir = os.path.join(jcqa_number_institute_dir, model)
        jcqa_katakana_model_dir = os.path.join(jcqa_katakana_institute_dir, model)
        number_results = load_results(os.path.join(jcqa_number_model_dir, "results.json"), by_choices=True)
        katakana_results = load_results(os.path.join(jcqa_katakana_model_dir, "results.json"), by_choices=True)
        agreements = []
        js_divergences = []
        katakana_result_dict = defaultdict(list)
        for katakana_result in katakana_results:
            katakana_result_dict[katakana_result.id].append(katakana_result)
        for number_result in number_results:
            katakana_result = [katakana_result for katakana_result in katakana_result_dict[number_result.id] if katakana_result.permutation == number_result.permutation][0]
            if np.linalg.norm(np.array(number_result.distributions) - np.array(katakana_result.distributions)) < 10**(-8):
                js_divergence = 0.0
            else:
                js_divergence = distance.jensenshannon(number_result.distributions, katakana_result.distributions)
            js_divergences.append(js_divergence)
            if np.argmax(np.array(number_result.distributions)) == np.argmax(np.array(katakana_result.distributions)):
                agreements.append(1)
            else:
                agreements.append(0)
        analyses[model] = {"js_divergence": np.mean(js_divergences), "accuracy": np.mean(agreements)}

analysis_dir = os.path.join("results/analysis", "jcqa_agreements_by_choices")
if not os.path.exists(analysis_dir):
    os.mkdir(analysis_dir)
write_analyses_json(analyses=analyses, file_path=os.path.join(analysis_dir, "results.json"))
write_analyses_csv(analyses=analyses, file_path=os.path.join(analysis_dir, "results.csv"))
print("jcqa")
print("all model")
print(np.mean([analyses[model]["accuracy"] for model in analyses]))
print("without stockmark")
print(np.mean([analyses[model]["accuracy"] for model in analyses if "stockmark" not in model]))
print("instruct")
print(np.mean([analyses[model]["accuracy"] for model in analyses if "chat" in model or "instruct" in model]))
print("base")
print(np.mean([analyses[model]["accuracy"] for model in analyses if "chat" not in model and "instruct" not in model]))
print("all model js")
print(np.mean([analyses[model]["js_divergence"] for model in analyses]))
print("without stockmark js")
print(np.mean([analyses[model]["js_divergence"] for model in analyses if "stockmark" not in model]))
print("instruct js")
print(np.mean([analyses[model]["js_divergence"] for model in analyses if "chat" in model or "instruct" in model]))
print("base js")
print(np.mean([analyses[model]["js_divergence"] for model in analyses if "chat" not in model and "instruct" not in model]))



institutes = os.listdir(v3_number_result_dir)
analyses = {}
for institute in institutes:
    v3_number_institute_dir = os.path.join(v3_number_result_dir, institute)
    v3_katakana_institute_dir = os.path.join(v3_katakana_result_dir, institute)
    models = list(sorted(os.listdir(v3_number_institute_dir)))
    for model in models:
        v3_number_model_dir = os.path.join(v3_number_institute_dir, model)
        v3_katakana_model_dir = os.path.join(v3_katakana_institute_dir, model)
        number_results = load_results(os.path.join(v3_number_model_dir, "results.json"), by_choices=True)
        katakana_results = load_results(os.path.join(v3_katakana_model_dir, "results.json"), by_choices=True)
        agreements = []
        js_divergences = []
        katakana_result_dict = defaultdict(list)
        for katakana_result in katakana_results:
            katakana_result_dict[katakana_result.id].append(katakana_result)
        for number_result in number_results:
            katakana_result = [katakana_result for katakana_result in katakana_result_dict[number_result.id] if katakana_result.permutation == number_result.permutation][0]
            if np.linalg.norm(np.array(number_result.distributions) - np.array(katakana_result.distributions)) < 10**(-8):
                js_divergence = 0.0
            else:
                js_divergence = distance.jensenshannon(number_result.distributions, katakana_result.distributions)
            js_divergences.append(js_divergence)
            if np.argmax(np.array(number_result.distributions)) == np.argmax(np.array(katakana_result.distributions)):
                agreements.append(1)
            else:
                agreements.append(0)
        analyses[model] = {"js_divergence": np.mean(js_divergences), "accuracy": np.mean(agreements)}

analysis_dir = os.path.join("results/analysis", "v3_agreements_by_choices")
if not os.path.exists(analysis_dir):
    os.mkdir(analysis_dir)
write_analyses_json(analyses=analyses, file_path=os.path.join(analysis_dir, "results.json"))
write_analyses_csv(analyses=analyses, file_path=os.path.join(analysis_dir, "results.csv"))
print("v3")
print("all model")
print(np.mean([analyses[model]["accuracy"] for model in analyses]))
print("without stockmark")
print(np.mean([analyses[model]["accuracy"] for model in analyses if "stockmark" not in model]))
print("instruct")
print(np.mean([analyses[model]["accuracy"] for model in analyses if "chat" in model or "instruct" in model]))
print("base")
print(np.mean([analyses[model]["accuracy"] for model in analyses if "chat" not in model and "instruct" not in model]))
print("all model js")
print(np.mean([analyses[model]["js_divergence"] for model in analyses]))
print("without stockmark js")
print(np.mean([analyses[model]["js_divergence"] for model in analyses if "stockmark" not in model]))
print("instruct js")
print(np.mean([analyses[model]["js_divergence"] for model in analyses if "chat" in model or "instruct" in model]))
print("base js")
print(np.mean([analyses[model]["js_divergence"] for model in analyses if "chat" not in model and "instruct" not in model]))
