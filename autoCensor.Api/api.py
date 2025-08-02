import os
import asyncio
from uuid import uuid4
from typing import Dict
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
from prepare_environment import EnvironmentManager
from inference import run_inference
from config import settings
from logger import get_logger
from pydantic import BaseModel, validator
import tempfile
import shutil

logger = get_logger(__name__)
app = FastAPI(title="AutoCensor API", version="1.0")

env_manager = EnvironmentManager.get_instance()

jobs: Dict[str, Dict] = {}

class PrepareResponse(BaseModel):
    message: str
    device: str
    video_path: str

class DetectRequest(BaseModel):
    tasks: str = "face,plate"
    conf: float = 0.25
    window_size: int = 5
    max_hold: int = 2
    tracker_cfg: str = "bytetrack.yaml"

    @validator("tasks")
    def validate_tasks(cls, v):
        allowed = {"face", "plate"}
        tasks_set = set(t.strip() for t in v.split(","))
        if not tasks_set.intersection(allowed):
            raise ValueError("tasks must include at least one of 'face' or 'plate'")
        return v

@app.post("/prepare", response_model=PrepareResponse)
async def prepare(
    video: UploadFile = File(...),
    face_model_path: str = Form(settings.MODEL_DIR + "/" + settings.FACE_MODEL_NAME),
    plate_model_path: str = Form(settings.MODEL_DIR + "/" + settings.PLATE_MODEL_NAME),
):
    contents = await video.read()
    if len(contents) > settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Uploaded video is too large")
    video.file.seek(0)

    video_path = env_manager.save_video(video)

    try:
        env_manager.load_models(face_model_path, plate_model_path)
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return PrepareResponse(
        message="Environment prepared successfully",
        device=env_manager.device,
        video_path=video_path,
    )

@app.post("/detect")
async def detect(request: DetectRequest, background_tasks: BackgroundTasks):
    if env_manager.video_path is None or not env_manager.models:
        raise HTTPException(status_code=400, detail="Environment is not prepared. Call /prepare first.")

    job_id = str(uuid4())
    progress = {"value": 0, "done": False, "result": None, "error": None}
    jobs[job_id] = progress

    def update_progress(p):
        progress["value"] = p

    def run():
        try:
            result = run_inference(
                env_manager.video_path,
                env_manager.models.get("face_model"),
                env_manager.models.get("plate_model"),
                [t.strip() for t in request.tasks.split(",")],
                request.conf,
                env_manager.device,
                request.window_size,
                request.max_hold,
                request.tracker_cfg,
                update_progress
            )
            progress["done"] = True
            progress["result"] = result
        except Exception as e:
            logger.error(f"Inference failed: {e}", exc_info=True)
            progress["done"] = True
            progress["error"] = str(e)

    background_tasks.add_task(run)

    return {"job_id": job_id}

@app.get("/progress/{job_id}")
def get_progress(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"progress": job["value"], "done": job["done"], "error": job["error"]}

@app.get("/result/{job_id}")
def get_result(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if not job["done"]:
        raise HTTPException(status_code=202, detail="Job still in progress")
    if job["error"]:
        raise HTTPException(status_code=500, detail=job["error"])
    return JSONResponse(content=job["result"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, log_level="info")


#uvicorn api:app --host 0.0.0.0 --port 8000 --reload