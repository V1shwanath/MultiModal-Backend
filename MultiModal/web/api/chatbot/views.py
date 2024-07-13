import logging
import shutil
import tempfile

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from fastapi.websockets import WebSocket, WebSocketDisconnect
from MultiModal.static.WhisperModel1 import whisper_model_instance
# from MultiModal.static.faster_whisper1 import transcription
# from MultiModal.static.phi3_visionchat import (
#     generate_response,
#     get_inputs,
#     reset_messages,
#     reset_img,
#     get_video_inputs,
# )
from MultiModal.static.phi3_visionchat import phi3_visionchat_instance as phi3_visionchat 
from MultiModal.static.video_inf import video_inf_instance as video_inf
# from MultiModal.static.video_inf import processing_status, video_to_frames
from MultiModal.static.vectordb import vector_store
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
    inputs = phi3_visionchat.get_inputs(image_bytes, text)
    try:
        answer = phi3_visionchat.generate_response(inputs)
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
    phi3_visionchat.reset_messages()
    phi3_visionchat.reset_img()
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
    transcription_text = whisper_model_instance.transcription(file.filename)
    return Response(content=transcription_text, media_type="application/json")

@router.post("/upload_video")
async def upload_video(video: UploadFile | None = None):
    video_inf.init_models()
    try:
        print("VIDEO FILE NAME:", video.filename)
        if video:
            print("TYPE ====",type(video))
            print("VIDEO FILE NAME:", video.filename)
            
            with tempfile.NamedTemporaryFile(delete=False) as temp_video:
                shutil.copyfileobj(video.file, temp_video)
            
            video_id = video.filename
            video_inf.processing_status[video_id] = "processing"
            video_inf.video_to_frames(temp_video.name, video_id)
            return {"video_id": video_id, "status": "processed"}
        else:
            # video_path = None
            print("VIDEO NOT FOUND")
    except Exception as e:
        print(f"ERROR: {e}")
        
    return {"answer": "SUCCESS"}


@router.post("/video_chatbot_2.0")
async def video_chatbot(text: str = Form(...), inference_type: str = Form(...), video_id: str = Form(...)):
    """
    Process text input through a video chatbot using the specified inference type.

    Args:
        text (str): The input text for the chatbot.
        inference_type (str): The type of inference to use ("Full Context" or "VectorDB Timestamp").

    Returns:
        dict: A dictionary with the chatbot's answer.
    """
    if video_id not in video_inf.processing_status:
        raise HTTPException(status_code=404, detail="Video not found")

    # Check if captions are available for the video
    if "captions" not in video_inf.processing_status[video_id]:
        raise HTTPException(
            status_code=400, detail="Captions not available for the video"
        )

    # Select the inference type
    if inference_type == "Full Context":
        inputs = phi3_visionchat.get_video_inputs(text, video_inf.processing_status[video_id]["captions"])
    elif inference_type == "VectorDB Timestamp":
        if (vector_store.text_embeddings is None):
            print("populatingggggggggggggggg")
            vector_store.populate_vectors(video_inf.processing_status[video_id]["captions"])
        results = vector_store.search_context(text)
        print("this is the results======================")
        inputs = phi3_visionchat.get_video_inputs(text, results)

        ## Clean up the collection to be added
    else:
        raise HTTPException(status_code=400, detail="Invalid inference type")

    # Generate response
    try:
        answer = phi3_visionchat.generate_response(inputs)
    except Exception as e:
        answer = f"ERROR: {e}"
        print(e)

    return {"answer": answer}

@router.get("/reset_chat_history")
def reset_history():
    phi3_visionchat.reset_messages()
    phi3_visionchat.reset_img()
    vector_store.delete_collection("my_collection")
    return "Message history wiped."


@router.get("/delete_model")
def delete_models():
    phi3_visionchat.delete_model()
    return "Model deleted"
