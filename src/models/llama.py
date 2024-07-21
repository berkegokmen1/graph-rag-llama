import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core import PromptTemplate


def get_llama_model(model_name, device):
    print("Using llm model:", model_name)

    system_prompt = "You are a Q&A assistant. Your goal is to answer questions as accurately as possible based on the instructions and context provided."

    # This will wrap the default prompts that are internal to llama-index
    query_wrapper_prompt = PromptTemplate("<|USER|>{query_str}<|ASSISTANT|>")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    stopping_ids = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    ]

    llm = HuggingFaceLLM(
        context_window=8192,
        max_new_tokens=64,
        generate_kwargs={"temperature": 0.7, "do_sample": False},
        system_prompt=system_prompt,
        query_wrapper_prompt=query_wrapper_prompt,
        tokenizer_name=model_name,
        model_name=model_name,
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
        stopping_ids=stopping_ids,
        tokenizer_kwargs={"max_length": 4096},
        # model_kwargs={"torch_dtype": torch.float16}
    )

    return llm, tokenizer
