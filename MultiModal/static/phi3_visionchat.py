import io
import os

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig

from MultiModal.settings import settings

os.environ["HF_HOME"] = settings.HF_HOME
# model_id = os.environ['MODEL_PATH']
model_id = "microsoft/Phi-3-vision-128k-instruct"
quant_config = BitsAndBytesConfig(load_in_4bit=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cuda",
    trust_remote_code=True,
    torch_dtype="auto",
    _attn_implementation="flash_attention_2",
    quantization_config=quant_config,
)

processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

messages = []
image_ref = None
IMAGE_PATH = r"..\images\prepped_img.jpg"


def update_messages(response):
    messages.append({"role": "assistant", "content": response})
    return messages


def generate_response(inputs):
    generation_args = {
        "max_new_tokens": 500,
        "temperature": 0.0,
        "do_sample": False,
    }

    generate_ids = model.generate(
        **inputs, eos_token_id=processor.tokenizer.eos_token_id, **generation_args
    )

    generate_ids = generate_ids[:, inputs["input_ids"].shape[1] :]
    response = processor.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    torch.cuda.empty_cache()
    update_messages(response)
    return response


def get_inputs(image_bytes, text):
    if image_bytes:
        prepped_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        prepped_img.save(IMAGE_PATH)
    else:
        if os.path.exists(IMAGE_PATH):
            prepped_img = Image.open(IMAGE_PATH).convert("RGB")
        else:
            prepped_img = None

    if image_bytes:
        reset_messages()
        messages.append({"role": "user", "content": f"<|image_1|>\n{text}"})
    else:
        messages.append({"role": "user", "content": f"{text}"})

    print("afer appending", messages)
    prompt = processor.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(prompt, prepped_img, return_tensors="pt").to("cuda:0")

    return inputs


def reset_messages():
    global messages
    torch.cuda.empty_cache()
    messages = []
    return messages


def reset_img():
    if os.path.isfile(IMAGE_PATH):
        os.remove(IMAGE_PATH)
