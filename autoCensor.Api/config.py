import os

from pydantic_settings import BaseSettings
class Settings(BaseSettings):
    MODEL_DIR: str = os.getenv("MODEL_DIR", "./models")
    VIDEO_DIR: str = os.getenv("VIDEO_DIR", "./videos")
    MAX_UPLOAD_SIZE_MB: int = 5000
    INFERENCE_TIMEOUT_SEC: int = 300

    FACE_MODEL_NAME: str = "face.pt"
    PLATE_MODEL_NAME: str = "license-plate.pt"

settings = Settings()
