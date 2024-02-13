from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, set_seed
import torch


def add_generation_prefix(tokenized_inputs: list[dict], tokenizer) -> list[dict]:
    empty_vocab_id = tokenizer.get_vocab()['▁']
    tokenized_inputs["input_ids"] = torch.cat((tokenized_inputs['input_ids'], torch.tensor([[empty_vocab_id]])), dim=1)
    tokenized_inputs["attention_mask"] = torch.cat((tokenized_inputs['attention_mask'], torch.tensor([[1]])), dim=1)
    return tokenized_inputs


def get_model_distributions(prompt, model, tokenizer, params, labels: list[str]) -> list[float]:
    with torch.no_grad():
        tokenized_inputs = tokenizer(prompt, return_tensors="pt")
        tokenized_inputs = add_generation_prefix(tokenized_inputs=tokenized_inputs, tokenizer=tokenizer)
        label_ids = [tokenizer.get_vocab()[label] for label in labels]
        outputs = model(**tokenized_inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        distributions = [predictions[0][-1][label_id] for label_id in label_ids]
    return distributions


def get_model_distributions_sft(prompt, model, tokenizer, params, labels: list[str]) -> list[float]:
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
