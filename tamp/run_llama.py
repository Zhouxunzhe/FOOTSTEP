import transformers
import torch
from huggingface_hub import login

# HF_ENDPOINT=https://hf-mirror.com python llama.py

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
hf_token = "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"
login(token=hf_token)

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.float16},
    device_map="auto",
)

overall_task = "You should pick cd and place it on strongbox."
scene_graph = """[cd is left to strongbox,
cd is down to strongbox,
cd is down to lamp,
strongbox is right to cd,
strongbox is up to cd,
strongbox is right to lamp,
strongbox is up to lamp,
lamp is up to cd,
lamp is left to strongbox,
lamp is down to strongbox]"""

messages = [
    {"role": "system", "content": "You are a server chatbot who will correct my action plan and REPLAN from the given task, scene graph, and the real scene output. You only need to output the replaned action and remember, the action plan are not executed yet!"},
    {"role": "user", "content": f"""
scene graph:{scene_graph}.
current scene:
[lamp: closed, cd: on the desk, strongbox: closed]
You can use function: (pickup, object), (goto, object), (put, object1, object2), (open, object), (close, object).
Here is your overall task: {overall_task}
Anction plan:
[(goto, lamp), (open, lamp), (pickup, cd), (goto, strongbox), (put, cd, strongbox)]
"""},
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
    temperature=0.6,
    top_p=0.9,
)
print(outputs[0]["generated_text"]) # [len(prompt):]
