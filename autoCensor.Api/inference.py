import cv2
import numpy as np
from collections import deque
from logger import get_logger

logger = get_logger(__name__)

def run_inference(
    video_path: str,
    face_model=None,
    plate_model=None,
    tasks=None,
    conf_threshold: float = 0.25,
    device: str = "cpu",
    window_size: int = 5,
    max_hold: int = 2,
    tracker_cfg: str = "bytetrack.yaml",
    progress_callback=None
):
    if tasks is None:
        tasks = ["face", "plate"]

    logger.info(f"Starting inference with tasks: {tasks}, video: {video_path}")

    cap_meta = cv2.VideoCapture(video_path)
    if not cap_meta.isOpened():
        raise RuntimeError(f"Cannot open video {video_path}")

    fps = cap_meta.get(cv2.CAP_PROP_FPS)
    width = int(cap_meta.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_meta.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap_meta.get(cv2.CAP_PROP_FRAME_COUNT))
    cap_meta.release()

    result = {
        "fps": fps,
        "width": width,
        "height": height,
        "frame_count": frame_count,
        "frames": {i: {"faces": [], "plates": []} for i in range(frame_count)}
    }

    # Face detection (no tracking)
    if "face" in tasks and face_model is not None:
        cap = cv2.VideoCapture(video_path)
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            detections = face_model(frame, conf=conf_threshold, device=device, verbose=False)
            boxes = detections[0].boxes if detections else None
            faces = []
            if boxes is not None:
                for box in boxes:
                    xyxy = box.xyxy.cpu().numpy().tolist()[0]
                    conf = float(box.conf.cpu())
                    faces.append({
                        "x1": int(xyxy[0]), "y1": int(xyxy[1]),
                        "x2": int(xyxy[2]), "y2": int(xyxy[3]),
                        "conf": round(conf, 3)
                    })
            result["frames"][frame_idx]["faces"] = faces

            # Progress güncelle
            if progress_callback:
                percent = int((frame_idx + 1) / frame_count * 50)  # yüzdelik olarak ilk 50'lik kısmı yüzüyor
                progress_callback(percent)

            frame_idx += 1
        cap.release()

    # Plate detection + tracking
    if "plate" in tasks and plate_model is not None:
        track_history = {}
        hold_counts = {}
        stream = plate_model.track(
            source=video_path,
            tracker=tracker_cfg,
            device=device,
            conf=conf_threshold,
            stream=True
        )
        for frame_idx, out in enumerate(stream):
            boxes = out.boxes
            current_ids = set()
            if boxes is not None and boxes.id is not None:
                xyxys = boxes.xyxy.cpu().numpy()
                ids = boxes.id.cpu().numpy().astype(int)
                for bb, tid in zip(xyxys, ids):
                    current_ids.add(int(tid))
                    if tid not in track_history:
                        track_history[tid] = deque(maxlen=window_size)
                        hold_counts[tid] = 0
                    track_history[tid].append(bb)
                    hold_counts[tid] = 0

            plates = []
            for tid, coords in track_history.items():
                if coords:
                    arr = np.array(coords)
                    x1, y1, x2, y2 = arr.mean(axis=0).astype(int)
                    plates.append({
                        "id": int(tid),
                        "x1": int(x1), "y1": int(y1),
                        "x2": int(x2), "y2": int(y2),
                        "conf": None
                    })

            result["frames"][frame_idx]["plates"] = plates

            # Progress güncelle
            if progress_callback:
                percent = 50 + int((frame_idx + 1) / frame_count * 50)  # ikinci yarısı
                progress_callback(min(percent, 100))

            # Update hold counts
            for tid in list(hold_counts.keys()):
                if tid not in current_ids:
                    hold_counts[tid] += 1
                    if hold_counts[tid] > max_hold:
                        del track_history[tid]
                        del hold_counts[tid]

    logger.info("Inference complete")
    return result
