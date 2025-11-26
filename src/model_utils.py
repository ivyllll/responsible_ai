# src/model_utils.py
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_mistral(model_name: str = "mistralai/Mistral-7B-Instruct-v0.1"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",         # uses GPU if available
        torch_dtype="auto",
        output_hidden_states=True  # IMPORTANT for our project
    )
    return model, tokenizer
