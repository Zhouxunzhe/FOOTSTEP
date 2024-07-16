import torch
import transformers

# HF_ENDPOINT=https://hf-mirror.com python tiny_llama.py
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

pipeline = transformers.pipeline(
    "text-generation", 
    model=model_id, 
    torch_dtype=torch.float16, 
    device_map="auto",
    )

messages = [
    {
        "role": "system",
        "content": "You are a friendly chatbot who always responds in the style of a pirate",
    },
    {"role": "user", "content": "How many helicopters can a human eat in one sitting?"},
]
prompt = pipeline.tokenizer.apply_chat_template(
    messages, 
    tokenize=False, 
    add_generation_prompt=True
    )

terminators = [
    pipeline.tokenizer.eos_token_id,
    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

outputs = pipeline(
    prompt, 
    max_new_tokens=256, 
    eos_token_id=terminators,
    do_sample=True, 
    temperature=0.7, 
    top_k=50, 
    top_p=0.95, 
    )
print(outputs[0]["generated_text"])
# <|system|>
# You are a friendly chatbot who always responds in the style of a pirate.</s>
# <|user|>
# How many helicopters can a human eat in one sitting?</s>
# <|assistant|>
# ...
