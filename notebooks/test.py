# from MultiModal.settings import settings
import gc
import json
import os

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

# os.environ["HF_HOME"] = settings.HF_HOME


class WhisperModel1:
    def __init__(self):
        pass

    def whisper_initalize(self, model_name="openai/whisper-large-v3"):
        self.whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_name,
            torch_dtype="auto",
            use_safetensors=True,
            attn_implementation="flash_attention_2",
            device_map="cuda:0",
        )
        # self.model = self.model.to("cuda:0")
        self.whisper_processor = AutoProcessor.from_pretrained(model_name)
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.whisper_model,
            tokenizer=self.whisper_processor.tokenizer,
            feature_extractor=self.whisper_processor.feature_extractor,
            max_new_tokens=128,
            chunk_length_s=30,
            batch_size=16,
            torch_dtype="auto",
        )

    def transcribe(self, audio_file):
        result = self.pipe(
            audio_file,
            generate_kwargs={"language": "english"},
            return_timestamps=True,
        )

        timestamps = result["chunks"]
        os.makedirs("../transcript_log", exist_ok=True)
        with open(
            r"media\transcription\transcription.json", "w", encoding="utf-8"
        ) as f:
            json.dump(timestamps, f, ensure_ascii=False, indent=4)
        return timestamps

    def delete_whisper_model(self):
        torch.cuda.empty_cache()
        del self.whisper_model
        del self.whisper_processor
        del self.pipe
        gc.collect()


whisper_model_instance = WhisperModel1()

whisper_model_instance.whisper_initalize()
print("Whisper model has been initialized")
whisper_model_instance.transcribe(r"media\audio.mp3")

whisper_model_instance.delete_whisper_model()

for i in range(1000000):
    print(i)
