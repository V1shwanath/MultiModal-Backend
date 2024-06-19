import logging
import shutil

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.websockets import WebSocket, WebSocketDisconnect

from MultiModal.static.faster_whisper1 import transcription
from MultiModal.static.phi3_visionchat import (
    generate_response,
    get_inputs,
    reset_messages,
)
from MultiModal.static.translation_demo import main

# Setup logger
logger = logging.getLogger("my_logger")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

router = APIRouter()

router.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


@router.get("/")
def index():
    """
    Returns a simple greeting message.
    """
    return "Hello World"


@router.post("/Chatbot")
async def opti_chatbot(text: str = Form(...), image: UploadFile = File(...) or None):
    """
    Handles chatbot interaction with text and optional image input.

    :param text: Text input from the user.
    :param image: Optional image input from the user.
    :return: JSON response containing the chatbot's answer.
    """
    try:
        logger.info(f"Received image of type: {type(image)}")
        image_bytes = await image.read()
    except Exception as err:
        logger.error(f"Error reading image: {err}")
        image_bytes = None

    inputs = get_inputs(image_bytes, text)
    try:
        answer = generate_response(inputs)
    except Exception as err:
        logger.error(f"Error generating response: {err}")
        answer = f"ERROR: {err}"

    return {"answer": answer}


@router.get("/reset_chat_history")
def reset_history():
    """
    Resets the chat history.

    :return: Confirmation message.
    """
    reset_messages()
    return "Message history wiped."


@router.websocket("/audio-stream")
async def audio_stream(websocket: WebSocket):
    """
    Handles audio stream via WebSocket connection.

    :param websocket: WebSocket connection.
    """
    await websocket.accept()
    logger.info("WebSocket connection established")

    try:
        while True:
            # Receive audio data in chunks
            data = await websocket.receive_bytes()
            logger.info(f"Received data chunk: {data}")
            response = main(data)
            logger.info(f"Transcription response: {response}")

            if not data:
                break
            await websocket.send_text(str(response))

    except WebSocketDisconnect:
        logger.info("WebSocket connection closed")
    except Exception as err:
        logger.error(f"Error in WebSocket connection: {err}")


@router.post("/upload/")
async def upload_video(text: str = Form(...), file: UploadFile = File(...)):
    """
    Handles video upload and transcription.

    :param text: Additional text input.
    :param file: Video file to be uploaded.
    :return: JSON response containing the transcription of the video.
    """
    with open(file.filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    transcription_text = transcription(file.filename)
    return Response(content=transcription_text, media_type="application/json")
