from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, set_seed
import torch
import openai
from src.settings import settings
from openai import OpenAI
import re


client = OpenAI(
    # This is the default and can be omitted
    api_key=settings.openai_api_key
)

SYSTEM_PROMPT = "あなたは世論調査に答えています。複数回答を指定された場合は、回答にあたる記号を複数出力してください。特に指定がない場合は、回答にあたる記号を一つだけ出力してください。"


def generate_model_distributions(prompt, model, tokenizer, params, labels: list[str]) -> list[float]:
    with torch.no_grad():
        tokenized_inputs = tokenizer(prompt, return_tensors="pt")
        tokenized_inputs = add_generation_prefix(tokenized_inputs=tokenized_inputs, tokenizer=tokenizer)
        label_ids = [tokenizer.get_vocab()[label] for label in labels]
        outputs = model(**tokenized_inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        distributions = [predictions[0][-1][label_id] for label_id in label_ids]
    return distributions


def generate_model_distributions_sft(prompt, model, tokenizer, params, labels: list[str]) -> list[float]:
    with torch.no_grad():
        messages = [{"role": "user", "content": prompt}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt += "回答:"
        tokenized_inputs = tokenizer(prompt, return_tensors="pt")
        tokenized_inputs = add_generation_prefix(tokenized_inputs=tokenized_inputs, tokenizer=tokenizer)
        label_ids = [tokenizer.get_vocab()[label] for label in labels]
        outputs = model(**tokenized_inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        distributions = [predictions[0][-1][label_id] for label_id in label_ids]
    return distributions


def generate_model_distributions_gpt(prompt, model, params, labels: list[str]) -> list[float]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
    print(prompt)
    while True:
        try:
            resp = client.chat.completions.create(model=model, messages=messages, **params)
            break
        except Exception as e:
            pass
    content = resp.choices[0].message.content
    print(content)
    return [1.0 if label in content else 0 for label in labels]
