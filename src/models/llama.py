import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def get_llama_model(path, device):
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path)

    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer
