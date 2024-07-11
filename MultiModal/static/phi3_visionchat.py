from PIL import Image
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from transformers import AutoProcessor
import torch
import io
import os
import gc

from MultiModal.settings import settings

# model_id = os.environ['MODEL_PATH']
# os.environ["HF_HOME"] = settings.HF_HOME
quant_config = BitsAndBytesConfig(load_in_4bit=True)

class phi3_visionchat:
    
    def __init__(self, model_id = "microsoft/Phi-3-vision-128k-instruct"):
        self.model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="cuda",
        trust_remote_code=True,
        torch_dtype="auto",
        _attn_implementation="flash_attention_2",
        quantization_config=quant_config,
        # cache_dir=r"..\HF_cache",
        )
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.messages = []
        self.image_ref = None
        self.IMAGE_PATH = r"..\images\prepped_img.jpg"
        
        
    def update_messages(self,response):
        self.messages.append({"role": "assistant", "content": response})
        return self.messages


    def generate_response(self,inputs):
        generation_args = {
            "max_new_tokens": 500,
            "temperature": 0.0,
            "do_sample": False,
        }

        generate_ids = self.model.generate(
            **inputs, eos_token_id=self.processor.tokenizer.eos_token_id, 
            **generation_args
        )

        generate_ids = generate_ids[:, inputs["input_ids"].shape[1]:]
        response = self.processor.batch_decode(
            generate_ids, skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]
        torch.cuda.empty_cache()
        self.update_messages(response)
        return response


    def get_inputs(self,image_bytes, text):
        if (image_bytes):
            prepped_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            prepped_img.save(self.IMAGE_PATH)
        else:
            if os.path.exists(self.IMAGE_PATH):
                prepped_img = Image.open(self.IMAGE_PATH).convert("RGB")
            else:
                prepped_img = None

        if image_bytes:
            self.reset_messages()
            self.messages.append({"role": "user", "content": f"<|image_1|>\n{text}"})
        else:
            self.messages.append({"role": "user", "content": f"{text}"})

        print("afer appending", self.messages)
        prompt = self.processor.tokenizer.apply_chat_template(
            self.messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(prompt, prepped_img, return_tensors="pt").to("cuda:0")

        return inputs

    def get_video_inputs(self,text, vid_context=None):
        if (self.messages == []):
            self.messages.append({"role": "user", "content": f"""

    System Prompt: Understanding Framewise Captions and Timestamps in a Video Sequence

    You are an advanced language model tasked with understanding and interpreting a sequence of events in a video. The data provided consists of framewise captions accompanied by timestamps. Each caption describes a specific scene or moment in the video at a precise time.

    Your objective is to accurately comprehend and organize these events to reflect the narrative flow of the video.

    Key points to note:

    Timestamps: Indicate the exact time in seconds when each frame appears in the video.
    Captions: Describe the visual content and context of each frame at the given timestamp.
    Sequence of Events:

    The events are presented in chronological order based on their timestamps.
    Each caption corresponds to a unique frame or scene in the video.
    The sequence builds a coherent narrative or story as the video progresses. \n

    {vid_context}\n Answer the Question: {text}"""})
        else:
            print("didnt get the Video Context")
            self.messages.append({"role": "user", "content": f"{text}"})

        print("afer appending", self.messages)
        prompt = self.processor.tokenizer.apply_chat_template(
            self.messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(prompt, return_tensors="pt").to("cuda:0")

        return inputs

    def reset_messages(self):
        global messages
        torch.cuda.empty_cache()
        self.messages = []
        return self.messages


    def reset_img(self):
        if os.path.isfile(self.IMAGE_PATH):
            os.remove(self.IMAGE_PATH)
            
    def delete_model(self):
        del self.model
        gc.collect()
        torch.cuda.empty_cache() 
        
phi3_visionchat_instance = phi3_visionchat()