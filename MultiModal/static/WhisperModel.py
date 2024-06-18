import os

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

from MultiModal.settings import settings

os.environ["HF_HOME"] = settings.HF_HOME


class WhisperModel:
    def __init__(
        self,
        model_id: str,
        torch_dtype: str = "auto",
        device: str = "cuda:0",
        use_safetensors: bool = True,
        attn_implementation: str = "flash_attention_2",
        files: dict = None,
    ):

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            use_safetensors=use_safetensors,
            attn_implementation=attn_implementation,
        )
        self.model = self.model.to(device)
        self.processor = AutoProcessor.from_pretrained(model_id)

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            max_new_tokens=128,
            chunk_length_s=30,
            batch_size=16,
            torch_dtype=torch_dtype,
            device=device,
        )

    def transcribe(self, audio_file):

        result = self.pipe(
            audio_file,
            generate_kwargs={"language": "english"},
            return_timestamps=True,
        )

        timestamps = result["chunks"]
        return timestamps
