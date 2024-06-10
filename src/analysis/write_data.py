import json
import csv


def write_analyses_json(analyses: dict, file_path) -> None:
    with open(file_path, "w") as fw:
        json.dump(analyses, fw)


def write_analyses_csv(analyses: dict, file_path) -> None:
    keys = sorted(list(list(analyses.values())[0].keys()))
    with open(file_path, "w") as fw:
        writer = csv.writer(fw)
        writer.writerow(["model"] + keys)
        for model in analyses:
            writer.writerow([model] + [analyses[model][key] for key in keys])
