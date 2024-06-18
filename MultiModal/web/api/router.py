from fastapi.routing import APIRouter

from MultiModal.web.api import auth, chatbot, monitoring

api_router = APIRouter()
api_router.include_router(monitoring.router)
api_router.include_router(chatbot.router, tags=["chatbot"], prefix="/chatbot")
api_router.include_router(auth.router, tags=["auth"], prefix="/auth")
