from typing import Any, Dict

import requests
from fastapi import APIRouter, Depends
from fastapi.security import OAuth2PasswordBearer
from jose import jwt

from MultiModal.settings import settings

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


router = APIRouter()


@router.get("/login/google")
def login_google() -> Dict[str, str]:
    return {
        "url": f"https://accounts.google.com/o/oauth2/auth?response_type=code&client_id={settings.google_client_id}&redirect_uri={settings.google_redirect_uri}&scope=openid%20profile%20email&access_type=offline",
    }


@router.get("/google")
async def auth_google(code: str) -> Dict[str, Any]:
    token_url = "https://accounts.google.com/o/oauth2/token"
    data = {
        "code": code,
        "client_id": settings.google_client_id,
        "client_secret": settings.google_client_secret,
        "redirect_uri": settings.google_redirect_uri,
        "grant_type": "authorization_code",
    }
    response = requests.post(token_url, data=data)
    response.raise_for_status()  # To ensure we raise an exception for bad responses
    access_token = response.json().get("access_token")
    user_info_response = requests.get(
        "https://www.googleapis.com/oauth2/v1/userinfo",
        headers={"Authorization": f"Bearer {access_token}"},
    )
    user_info_response.raise_for_status()  # To ensure we raise an exception for bad responses
    return user_info_response.json()


@router.get("/token")
async def get_token(token: str = Depends(oauth2_scheme)) -> Dict[str, Any]:
    return jwt.decode(token, settings.google_client_secret, algorithms=["HS256"])
