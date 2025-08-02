import os
import shutil
from pathlib import Path
from ultralytics import YOLO
import torch
from threading import Lock
from config import settings
from logger import get_logger

logger = get_logger(__name__)

class EnvironmentManager:
    _instance = None
    _lock = Lock()

    def __init__(self):
        self.video_path = None
        self.models = {}
        self.device = self._detect_device()

    @classmethod
    def get_instance(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def _detect_device(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        return device

    def save_video(self, upload_file) -> str:
        # Ensure video dir exists
        Path(settings.VIDEO_DIR).mkdir(parents=True, exist_ok=True)
        safe_filename = os.path.basename(upload_file.filename)
        file_path = os.path.join(settings.VIDEO_DIR, safe_filename)

        with open(file_path, "wb") as f:
            shutil.copyfileobj(upload_file.file, f)

        self.video_path = file_path
        logger.info(f"Saved uploaded video to {file_path}")
        return file_path

    def load_models(self, face_model_path: str = None, plate_model_path: str = None):
        # Load face model
        if face_model_path:
            face_path = Path(face_model_path)
            if not face_path.is_file():
                raise FileNotFoundError(f"Face model not found at {face_model_path}")
            self.models["face_model"] = YOLO(str(face_path))
            self.models["face_model"].to(self.device)
            logger.info(f"Loaded face model from {face_model_path}")
        else:
            self.models["face_model"] = None

        # Load plate model
        if plate_model_path:
            plate_path = Path(plate_model_path)
            if not plate_path.is_file():
                raise FileNotFoundError(f"Plate model not found at {plate_model_path}")
            self.models["plate_model"] = YOLO(str(plate_path))
            self.models["plate_model"].to(self.device)
            logger.info(f"Loaded plate model from {plate_model_path}")
        else:
            self.models["plate_model"] = None

    def reset(self):
        self.video_path = None
        self.models = {}
        logger.info("Environment reset")

