# JAPANION QA SURVEY

## Environment Construction
```bash
docker build --platform linux/amd64 -t japanion_qa -f ./Dockerfile .
```


## Running example

```bash
docker run -it japanion_qa /bin/bash
poetry run python src/run.py \
  --language_model HuggingFaceLM \
  --language_model.model_name "tokyotech-llm/Swallow-13b-instruct-v0.1" \
  --save_dir "results/request/Swallow-13b-instruct-v0.1" \
  --input "sbintuitions/japanion_qa"
```
