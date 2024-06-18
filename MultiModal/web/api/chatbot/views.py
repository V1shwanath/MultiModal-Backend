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
    return "Hello World"


@router.post("/Chatbot")
async def opti_chatbot(text: str = Form(...), image: UploadFile = File(...) or None):
    try:
        print(type(image))
        image_bytes = await image.read()
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
    reset_messages()
    return "Message history wiped."


@router.websocket("/audio-stream")
async def audio_stream(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket connection established")

    try:
        while True:
            # Receive audio data in chunks
            data = await websocket.receive_bytes()
            print(data)
            a = main(data)
            print(str(a))
            if not data:
                break
            await websocket.send_text(str(a))

    except WebSocketDisconnect:
        print("WebSocket connection closed")
    except Exception as e:
        print(f"Error in WebSocket connection: {e}")


@router.post("/upload/")
async def upload_video(text: str = Form(...), file: UploadFile = File(...)):
    with open(file.filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    text = transcription(file.filename)

    return Response(content=text, media_type="application/json")
