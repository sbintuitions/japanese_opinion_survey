# JAPANION OPINION SURVEY

## Summary
Code for investigating the opinions of large language models with multiple-choice QA format.

### Data
- [JapanionQA](https://huggingface.co/datasets/sbintuitions/japanion_qa)
- [Attributes Data for JapanionQA](https://huggingface.co/datasets/sbintuitions/japanion_qa_attributes)

## Docker Image Construction
```bash
docker build --platform linux/amd64 -t japanion_qa -f ./Dockerfile .
```


## Running example

```bash
docker run -it japanion_qa /bin/bash
poetry run python src/run.py \
  --language_model HuggingFaceLM \
  --language_model.model_name "sbintuitions/sarashina1-7b" \
  --save_dir "results/request/sbintuitions/sarashina1-7b" \
  --input "sbintuitions/japanion_qa"
```
