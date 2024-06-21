import logging
import shutil

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import Response
from fastapi.websockets import WebSocket, WebSocketDisconnect

from MultiModal.static.faster_whisper1 import transcription
from MultiModal.static.phi3_visionchat import (
    generate_response,
    get_inputs,
    reset_messages,
    reset_img,
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


@router.get("/")
def index():
    """
    Returns a simple greeting message.
    """
    return "Hello World"


@router.post("/Chatbot")
async def opti_chatbot(text: str = Form(...), image: UploadFile | None = None):
    """
    Handles chatbot requests.
    Accepts text and image inputs and returns a response.

    :param text: Text input.
    :param image: Image input.
    """
    try:
        if image:
            print(type(image))
            image_bytes = await image.read()
        else:
            image_bytes = None
    except Exception as e:
        print(f"ERROR: {e}")
    inputs = get_inputs(image_bytes, text)
    try:
        answer = generate_response(inputs)
    except Exception as e:
        answer = f"ERROR: {e}"
        print(e)
    return {"answer": answer}


@router.get("/reset_chat_history")
def reset_history():
    """
    Resets chat history.
    :param request: Request object.
    """
    reset_messages()
    reset_img()
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
