import argparse
import io
from queue import Queue
from sys import platform
from tempfile import NamedTemporaryFile

import nltk
import speech_recognition as sr
from pydub import AudioSegment
from WhisperModel import WhisperModel


def main(data_incoming):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="large",
        help="Model to use",
        choices=["tiny", "base", "small", "medium", "large"],
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="device to use for CTranslate2 inference",
        choices=["auto", "cuda", "cpu"],
    )
    parser.add_argument(
        "--compute_type",
        default="auto",
        help="Type of quantization to use",
        choices=[
            "auto",
            "int8",
            "int8_floatt16",
            "float16",
            "int16",
            "float32",
        ],
    )
    parser.add_argument(
        "--translation_lang",
        default="English",
        help="Which language should we translate into.",
        type=str,
    )
    parser.add_argument(
        "--non_english",
        action="store_true",
        help="Don't use the English model.",
    )
    parser.add_argument(
        "--threads",
        default=0,
        help="number of threads used for CPU inference",
        type=int,
    )
    parser.add_argument(
        "--energy_threshold",
        default=1000,
        help="Energy level for mic to detect.",
        type=int,
    )
    parser.add_argument(
        "--record_timeout",
        default=2,
        help="How real time the recording is in seconds.",
        type=float,
    )
    parser.add_argument(
        "--phrase_timeout",
        default=3,
        help="How much empty space between recordings before we "
        "consider it a new line in the transcription.",
        type=float,
    )

    if "linux" in platform:
        parser.add_argument(
            "--default_microphone",
            default="pulse",
            help="Default microphone name for SpeechRecognition. "
            "Run this with 'list' to view available Microphones.",
            type=str,
        )
    args = parser.parse_args()

    last_sample = bytes()
    data_queue = Queue()
    recorder = sr.Recognizer()
    recorder.energy_threshold = args.energy_threshold
    recorder.dynamic_energy_threshold = False

    source = None

    if "linux" in platform:
        mic_name = args.default_microphone
        if not mic_name or mic_name == "list":
            print("Available microphone devices are: ")
            for index, name in enumerate(
                sr.Microphone.list_microphone_names(),
            ):
                print(f'Microphone with name "{name}" found')
                return
        else:
            for index, name in enumerate(
                sr.Microphone.list_microphone_names(),
            ):
                if mic_name in name:
                    source = sr.Microphone(
                        sample_rate=16000,
                        device_index=index,
                    )
                    break
            if source is None:
                print(f'Microphone with name "{mic_name}" not found.')
                return
    else:
        source = sr.Microphone(sample_rate=16000)

    print(type(source))
    print(source)

    if args.model == "large":
        args.model = "openai/whisper-large-v3"

    model = args.model

    nltk.download("punkt")
    print(model)
    audio_model = WhisperModel(model)

    temp_file = NamedTemporaryFile().name
    transcription = [""]

    source = data_incoming

    data_queue.put(data_incoming)
    # with source:
    #     recorder.adjust_for_ambient_noise(source)

    # def record_callback(_, audio: sr.AudioData) -> None:
    #     data = audio.get_raw_data(

    #     )
    #     print("the raw data is",type(data))
    #     data_queue.put(data)

    # recorder.listen_in_background(source, record_callback,
    # phrase_time_limit=record_timeout
    # )

    print("Model loaded.\n")

    while True:
        try:
            if not data_queue.empty():
                while not data_queue.empty():
                    data = data_queue.get()
                    last_sample += data
                audio = AudioSegment.from_file(io.BytesIO(data), format="webm")
                wav_io = io.BytesIO()
                audio.export(wav_io, format="wav")

                with open(temp_file, "w+b") as f:
                    f.write(wav_io.read())

                result = audio_model.transcribe(temp_file)
                return result

        except KeyboardInterrupt:
            break

    print("\n\nTranscription:")
    for line in transcription:
        print(line)


def update_text(sentences, translation_lang):
    print("Transcription:")
    for sentence in sentences:
        print(sentence)
