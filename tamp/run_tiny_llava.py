from PIL import Image
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

# HF_ENDPOINT=https://hf-mirror.com python tiny_llava.py

model_id = "bczhou/tiny-llava-v1-hf"

processor = AutoProcessor.from_pretrained(
    model_id
    )

model = LlavaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map='auto',
    )

# prepare image and text prompt, using the appropriate prompt template
image_path = "./llava_v1_5_radar.jpg"
image = Image.open(image_path)
prompt = "[INST] <image>\nWhat is shown in this image? [/INST]"

inputs = processor(prompt, image, return_tensors='pt')

# autoregressively complete prompt
output = model.generate(**inputs, max_new_tokens=1000, do_sample=False)

print(processor.decode(output[0], skip_special_tokens=True))
