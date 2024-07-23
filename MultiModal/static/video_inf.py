import gc
import json
import os
from pprint import pprint
from typing import Dict

import cv2
import numpy as np
import requests
import torch
from moviepy.editor import VideoFileClip
from PIL import Image
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    ViTImageProcessor,
    ViTModel,
)

from MultiModal.static.WhisperModel1 import whisper_model_instance

# florence_model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-base", device_map='cuda', trust_remote_code=True,torch_dtype = torch.bfloat16, attn_implementation="flash_attention_2")
# flor_processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True)
# processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
# model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')


class video_inf:
    def __init__(self):
        self.prompt = "<MORE_DETAILED_CAPTION>"
        self.processing_status: Dict[str, str] = {}

    def init_models(
        self,
        florence_model_name="microsoft/Florence-2-base",
        vit_model_name="google/vit-base-patch16-224-in21k",
    ):
        self.florence_model = AutoModelForCausalLM.from_pretrained(
            florence_model_name,
            device_map="cuda",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        self.flor_processor = AutoProcessor.from_pretrained(
            florence_model_name, trust_remote_code=True
        )
        self.processor = ViTImageProcessor.from_pretrained(vit_model_name)
        self.model = ViTModel.from_pretrained(vit_model_name)

    def get_inf(self, path):
        torch.cuda.empty_cache()
        image = Image.open(path).convert("RGB")
        inputs = self.flor_processor(
            text=self.prompt, images=image, return_tensors="pt"
        ).to("cuda")
        inputs = inputs.to(torch.bfloat16)

        generated_ids = self.florence_model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3,
            do_sample=False,
        )
        generated_text = self.flor_processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]

        parsed_answer = self.flor_processor.post_process_generation(
            generated_text, task=self.prompt, image_size=(image.width, image.height)
        )

        return str(parsed_answer["<MORE_DETAILED_CAPTION>"])

    def img_embeddings(self, img1):
        image1 = Image.open(img1).convert("RGB")
        inputs1 = self.processor(images=image1, return_tensors="pt")
        outputs1 = self.model(**inputs1)
        last_hidden_states1 = outputs1.last_hidden_state
        return last_hidden_states1

    def similarity(self, img_embeddings1, img_embeddings2):
        img_embeddings1 = img_embeddings1.flatten()
        img_embeddings2 = img_embeddings2.flatten()
        sim = torch.nn.functional.cosine_similarity(
            img_embeddings1, img_embeddings2, dim=0
        )
        print("this is the similairity", sim)
        return sim

    def save_frame(self, frame, filename):
        cv2.imwrite(filename, frame)
        # print(f"Saved {filename}")
        return filename
    
    def adjust_timestamps(self,transcripts):
        adjusted_transcripts = []
        offset = 0.0
        for i in range(len(transcripts)):
            if i > 0 and transcripts[i]['timestamp'][0] < transcripts[i-1]['timestamp'][0]:
                offset += 30.0  # Add 30 seconds offset when timestamp resets
            adjusted_transcripts.append({
                'timestamp': (transcripts[i]['timestamp'][0] + offset, transcripts[i]['timestamp'][1] + offset),
                'text': transcripts[i]['text']
            })
        return adjusted_transcripts

    def video_to_frames(
        self, video_path, video_id, output_folder=r"..\videos\frames", frame_interval=25
    ):
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
                frame_filename = os.path.join(
                    output_folder, f"frame_{saved_frame_num:04d}.jpg"
                )
                self.save_frame(frame, frame_filename)
                if previous_embedding is None:
                    previous_embedding = self.img_embeddings(frame_filename)
                    caption = self.get_inf(frame_filename)
                    results.append({"timestamp": "0s", "caption": caption})
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
                        results.append({"timestamp": timestamp, "caption": caption})
                        saved_frame_num += 1
                print(caption)
            frame_num += frame_interval

        # Release the video capture object
        cap.release()
        print(f"Extracted {saved_frame_num} frames to {output_folder}")
        torch.cuda.empty_cache()
        del self.florence_model
        del self.model
        del self.flor_processor
        del self.processor
        torch.cuda.empty_cache()
        gc.collect()

        # print(results)
        # open log/transcript.json
        video = VideoFileClip(video_path)
        audio = video.audio
        audio_path = r"audio.wav"
        audio.write_audiofile("audio.wav")
        whisper_model_instance.whisper_initalize()
        transcriptions = whisper_model_instance.transcribe("audio.wav")
        corrected_transcriptions = self.adjust_timestamps(transcriptions)
        print('-------------------------------------------------')
        print(corrected_transcriptions)
        print('-------------------------------------------------')
        print(results)
        print('-------------------------------------------------')
        # video_transcript_path = r"media\transcription\transcription.json"
        # with open(video_transcript_path, "r") as file:
        #     video_transcript_data = json.load(file)
        video_transcript_data = corrected_transcriptions
        frame_captions_data = results
        # add captions to the transcript
        merged_data = []
        used_transcripts = set()

        for caption in frame_captions_data:
            caption_time = self.timestamp_to_seconds(caption["timestamp"])
            matched = False
            for transcript in video_transcript_data:
                start, end = transcript["timestamp"]
                if (
                    start <= caption_time <= end
                    and transcript["text"] not in used_transcripts
                ):
                    merged_data.append(
                        {
                            "timestamp": caption["timestamp"],
                            "caption": caption["caption"],
                            "transcription": transcript["text"],
                        }
                    )
                    used_transcripts.add(transcript["text"])
                    matched = True
                    break
            if not matched:
                merged_data.append(
                    {
                        "timestamp": caption["timestamp"],
                        "caption": caption["caption"],
                        'transcription': ''
                    }
                )

        print("=============", merged_data)
        self.processing_status[video_id] = {
            "status": "complete",
            "captions": merged_data,
        }

        whisper_model_instance.delete_whisper_model()
        # self.processing_status[video_id] = {"status": "complete", "captions" : results}
        # print(merged_data)
        print("Processing complete. ==========================")

        return merged_data
        # return results

    def delete_model(self):
        torch.cuda.empty_cache()
        del self.florence_model
        del self.model
        del self.flor_processor
        del self.processor
        gc.collect()
        torch.cuda.empty_cache()

    def reset_processing_status(self):
        self.processing_status.clear()
        return "Processing status reset."

    def timestamp_to_seconds(self, timestamp):
        if "m" in timestamp and "s" in timestamp:
            minutes, seconds = timestamp.split("m")
            seconds = seconds.replace("s", "")
            return int(minutes) * 60 + float(seconds)
        elif "s" in timestamp:
            seconds = timestamp.replace("s", "")
            return float(seconds)
        return 0


video_inf_instance = video_inf()

# video_path = r'C:\Users\jvish\OneDrive\Documents\VISH_Stuff\vw-multimodal-backend\vw-MultiModalAI-backend\src\videos\video.mp4'
# output_folder = r'C:\Users\jvish\OneDrive\Documents\VISH_Stuff\vw-multimodal-backend\vw-MultiModalAI-backend\src\videos\frames'
# frame_interval = 75

# captions = video_to_frames(video_path, output_folder, frame_interval)
# pprint(captions)
