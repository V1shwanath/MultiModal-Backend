import requests
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM ,ViTImageProcessor, ViTModel
import numpy as np
import os
import cv2
from pprint import pprint
from typing import Dict
import json
import gc


# florence_model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-base", device_map='cuda', trust_remote_code=True,torch_dtype = torch.bfloat16, attn_implementation="flash_attention_2")
# flor_processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True)
# processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
# model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')


class video_inf:
    
    def __init__(self):
        self.prompt = "<MORE_DETAILED_CAPTION>"
        self.processing_status: Dict[str, str] = {}
        
    def init_models(self, florence_model_name="microsoft/Florence-2-base", vit_model_name='google/vit-base-patch16-224-in21k'):
        self.florence_model = AutoModelForCausalLM.from_pretrained(
            florence_model_name, 
            device_map='cuda', 
            trust_remote_code=True, 
            torch_dtype=torch.bfloat16, 
            attn_implementation="flash_attention_2"
        )
        self.flor_processor = AutoProcessor.from_pretrained(florence_model_name, trust_remote_code=True)
        self.processor = ViTImageProcessor.from_pretrained(vit_model_name)
        self.model = ViTModel.from_pretrained(vit_model_name)

    def get_inf(self,path):
        torch.cuda.empty_cache()
        image = Image.open(path).convert("RGB")
        inputs = self.flor_processor(text=self.prompt, images=image, return_tensors="pt").to("cuda")
        inputs = inputs.to(torch.bfloat16)

        generated_ids = self.florence_model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3,
            do_sample=False
        )
        generated_text = self.flor_processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

        parsed_answer = self.flor_processor.post_process_generation(generated_text, task=self.prompt, image_size=(image.width, image.height))

        return str(parsed_answer['<MORE_DETAILED_CAPTION>'])

    def img_embeddings(self,img1):
        image1 = Image.open(img1).convert("RGB")
        inputs1 = self.processor(images=image1, return_tensors="pt")
        outputs1 = self.model(**inputs1)
        last_hidden_states1 = outputs1.last_hidden_state
        return last_hidden_states1

    def similarity(self,img_embeddings1, img_embeddings2):
        img_embeddings1 = img_embeddings1.flatten()
        img_embeddings2 = img_embeddings2.flatten()
        sim = torch.nn.functional.cosine_similarity(img_embeddings1 ,img_embeddings2 ,dim=0)
        print("this is the similairity" , sim)
        return sim

    def save_frame(self,frame, filename):
        cv2.imwrite(filename, frame)
        # print(f"Saved {filename}")
        return filename


    def video_to_frames(self,video_path,video_id, output_folder=r"..\videos\frames", frame_interval=75 ):
        """
        Extract frames from a video at a specified interval and process them with get_inf.

        Parameters:
        - video_path: str, path to the video file.
        - output_folder: str, folder where the frames will be saved.
        - frame_interval: int, save every nth frame.
        """
        # Check if output folder exists, if not, create it
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Open the video file
        cap = cv2.VideoCapture(video_path)

        # Check if the video was opened successfully
        if not cap.isOpened():
            print("Error: Could not open video.")
            return

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = frame_count / fps

        print(f"Total frames: {frame_count}")

        frame_num = 0
        saved_frame_num = 0
        results = []
        previous_embedding = None

        while True:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)

            ret, frame = cap.read()

            if not ret:
                break
            if frame_num % frame_interval == 0:
                frame_filename = os.path.join(output_folder, f"frame_{saved_frame_num:04d}.jpg")
                self.save_frame(frame, frame_filename)
                if previous_embedding is None:
                    previous_embedding = self.img_embeddings(frame_filename)
                    caption = self.get_inf(frame_filename)
                    results.append({'timestamp':"0s", 'caption':caption})
                    saved_frame_num += 1
                else:
                    current_embedding = self.img_embeddings(frame_filename)
                    sim = self.similarity(previous_embedding, current_embedding)
                    if sim > 0.9:
                        frame_num += frame_interval
                        continue
                    else:
                        previous_embedding = current_embedding
                        caption = self.get_inf(frame_filename)
                        timestamp = f"{int(frame_num/fps/60)}m {frame_num/fps%60}s"
                        results.append({'timestamp':timestamp, 'caption':caption})
                        saved_frame_num += 1
                print(caption)
            frame_num += frame_interval

        # Release the video capture object
        cap.release()
        print(f"Extracted {saved_frame_num} frames to {output_folder}")
        self.processing_status[video_id] = {"status": "complete", "captions" : results}
        torch.cuda.empty_cache()
        del self.florence_model
        del self.model
        del self.flor_processor
        del self.processor

        gc.collect()
        torch.cuda.empty_cache() 

        print(results)
        return results
    def reset_processing_status(self):
        self.processing_status.clear()
        return "Processing status reset."

video_inf_instance = video_inf()

    # video_path = r'C:\Users\jvish\OneDrive\Documents\VISH_Stuff\vw-multimodal-backend\vw-MultiModalAI-backend\src\videos\video.mp4'
    # output_folder = r'C:\Users\jvish\OneDrive\Documents\VISH_Stuff\vw-multimodal-backend\vw-MultiModalAI-backend\src\videos\frames'
    # frame_interval = 75

    # captions = video_to_frames(video_path, output_folder, frame_interval)
    # pprint(captions)