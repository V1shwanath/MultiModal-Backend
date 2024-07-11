import json
import os
import time
import warnings

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

from MultiModal.settings import settings

# os.environ["HF_HOME"] = settings.HF_HOME


def transcription(file):
    warnings.filterwarnings("ignore")
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id = "openai/whisper-large-v3"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
        _attn_implementation="flash_attention_2",
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=16,
        return_timestamps=True,
        torch_dtype=torch_dtype,
        generate_kwargs={"language": "english"},
        model_kwargs={"attn_implementation": "flash_attention_2"},
        device=device,
    )

    start_time = time.time()

    # Perform the transcription
    result = pipe(file)

    print(
        "Detected language '%s' with probability %f"
        % (result["language"], result["language_probability"]),
    )

    # Collect segments details
    segment_details = []
    for segment in result["segments"]:
        segment_info = {
            "start": segment["start"],
            "end": segment["end"],
            "text": segment["text"],
        }
        segment_details.append(segment_info)

    end_time = time.time()
    total_time = end_time - start_time

    # Create a dictionary for the JSON response
    response = {
        "transcription": result["text"],
        "language": result["language"],
        "language_probability": result["language_probability"],
        "segments": segment_details,
        "total_time": total_time,
    }

    # Convert the dictionary to a JSON object
    response_json = json.dumps(response, indent=4)
    print(response_json)

    return response_json
