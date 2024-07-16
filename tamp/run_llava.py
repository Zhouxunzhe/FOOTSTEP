from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image
import requests

# HF_ENDPOINT=https://hf-mirror.com python llava.py

model_id = "llava-hf/llava-v1.6-mistral-7b-hf"

processor = LlavaNextProcessor.from_pretrained(
    model_id
    )

model = LlavaNextForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True,
    device_map='auto',
    )

# prepare image and text prompt, using the appropriate prompt template
image_path = "./RGB_annotated.jpg"
image = Image.open(image_path)

prompt = """[INST] <image>
scene graph: 
[cd is left to strongbox,
cd is down to strongbox,
cd is down to lamp,
strongbox is right to cd,
strongbox is up to cd,
strongbox is right to lamp,
strongbox is up to lamp,
lamp is up to cd,
lamp is left to strongbox,
lamp is down to strongbox]
Accoring to the scene graph and image input, please generate a task_list with functions as the action sequence.
You can use functions: (pickup, object), (goto, object), (put, object1, object2), (open, object), (close, object).
Here is your overall task:
You should go to pick up cd and place it on strongbox.
[/INST]"""

inputs = processor(prompt, image, return_tensors="pt")

# autoregressively complete prompt
output = model.generate(**inputs, max_new_tokens=100)

print(processor.decode(output[0], skip_special_tokens=True))
