import subprocess
import shutil
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import os
import requests
import tempfile

from audio import analyze_audio
from baseline import get_baseline, update_baseline
from config import MODEL_VERSION
from logger import get_logger
from ml.features import features_from_signals, vector_from_features
from ml.runtime import MLRuntime
from scoring import compute_result, compute_task_score
from utils import is_url, remove_temp_file
from video import analyze_video
from vision import analyze_face


# =========================
# App & Runtime
# =========================

app = FastAPI()

@app.get("/")
def root():
    return {"status": "ok"}

logger = get_logger()

ml_runtime = MLRuntime()
ml_runtime.load()

DIRECTUS_URL = os.getenv("DIRECTUS_URL")
DIRECTUS_TOKEN = os.getenv("DIRECTUS_TOKEN")


# =========================
# Models
# =========================

class Media(BaseModel):
    image: str | None = Field(None, description="Local path, URL, or Directus asset ID")
    audio: str | None = Field(None, description="Local path, URL, or Directus asset ID")
    video: str | None = Field(None, description="Local path, URL, or Directus asset ID")


class Task(BaseModel):
    reaction_time: float | None = None
    errors: int | None = None


class ScanRequest(BaseModel):
    scan_id: str
    media: Media
    task: Task | None = None
    previous_confidence: float | None = None
    subject_id: str | None = None


class BaselineRequest(BaseModel):
    subject_id: str
    media: Media
    task: Task | None = None


class SignalDiagnostics(BaseModel):
    score: float | None = None
    details: dict = Field(default_factory=dict)


class Diagnostics(BaseModel):
    missing_media: list[str]
    signals: dict[str, SignalDiagnostics]
    confidence_components: dict
    baseline: dict | None = None
    ml: dict | None = None


class ScanResponse(BaseModel):
    overall_state: str
    confidence: float
    confidence_drift: float
    baseline_confidence: float | None = None
    baseline_drift: float | None = None
    camera_confidence: float
    video_confidence: float
    voice_confidence: float
    task_performance_score: int
    task_score: float | None = None
    missing_media: list[str]
    alerts: list[str]
    explanation: str
    medical_report: str
    confidence_components: dict
    diagnostics: Diagnostics | None = None
    model_version: str


class BaselineResponse(BaseModel):
    subject_id: str
    baseline: dict
    model_version: str

# =========================
# Helpers
# =========================

def _convert_audio_to_wav(path: str) -> str:
    if shutil.which("ffmpeg") is None:
        raise HTTPException(status_code=500, detail="ffmpeg not installed")
    fd, out = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    cmd = ["ffmpeg", "-y", "-i", path, "-ac", "1", "-ar", "16000", out]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return out

def _download_directus_asset(asset_id: str, suffix: str) -> str:
    if not DIRECTUS_URL or not DIRECTUS_TOKEN:
        raise HTTPException(
            status_code=500,
            detail="Directus credentials are not configured"
        )

    url = f"{DIRECTUS_URL}/assets/{asset_id}"
    headers = {"Authorization": f"Bearer {DIRECTUS_TOKEN}"}

    try:
        r = requests.get(url, headers=headers, timeout=30)
        r.raise_for_status()
    except Exception:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to download media: {asset_id}"
        )

    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(r.content)

    return path



def _resolve_media_input(value: str | None, suffix: str):
    if not value:
        return None, False

    # URL مباشر
    if is_url(value):
        try:
            r = requests.get(value, timeout=30)
            r.raise_for_status()
        except Exception:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to download media: {value}"
            )

        fd, path = tempfile.mkstemp(suffix=suffix)
        with os.fdopen(fd, "wb") as f:
            f.write(r.content)

        return path, True

    # Directus asset ID
    if DIRECTUS_URL and DIRECTUS_TOKEN:
        path = _download_directus_asset(value, suffix)
        return path, True

    # Local file
    if os.path.exists(value):
        return value, False

    raise HTTPException(
        status_code=400,
        detail=f"Media file not found: {value}"
    )


def _safe_analyze(fn, path):
    if not path:
        return {"score": None, "details": {"status": "missing"}}
    try:
        result = fn(path)
    except Exception:
        return {"score": None, "details": {"status": "error"}}
    if isinstance(result, dict) and "score" in result:
        return result
    return {"score": result, "details": {}}


def _ensure_media_present(media: Media) -> None:
    if not any([media.image, media.audio, media.video]):
        raise HTTPException(
            status_code=422,
            detail="At least one media input is required."
        )


def _analyze_media(media: Media):
    _ensure_media_present(media)
    temp_files = []

    image_path, is_temp = _resolve_media_input(media.image, ".jpg")
    if is_temp:
        temp_files.append(image_path)

    audio_path, is_temp = _resolve_media_input(media.audio, ".bin")
if is_temp:
    temp_files.append(audio_path)

if audio_path:
    try:
        converted = _convert_audio_to_wav(audio_path)
        temp_files.append(converted)
        audio_path = converted
    except Exception:
        return {"camera": camera, "video": video, "voice": {"score": None, "details": {"status": "load_failed"}}}, temp_files

    video_path, is_temp = _resolve_media_input(media.video, ".mp4")
    if is_temp:
        temp_files.append(video_path)

    camera = _safe_analyze(analyze_face, image_path)
    video = _safe_analyze(analyze_video, video_path)
    voice = _safe_analyze(analyze_audio, audio_path)

    return {"camera": camera, "video": video, "voice": voice}, temp_files


# =========================
# Routes
# =========================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_version": MODEL_VERSION,
        "ml_loaded": ml_runtime.is_loaded(),
        "ml_error": ml_runtime.error,
    }


@app.post("/baseline", response_model=BaselineResponse)
def set_baseline(req: BaselineRequest):
    signals, temp_files = _analyze_media(req.media)
    try:
        logger.info("baseline_update subject_id=%s", req.subject_id)

        scores = {k: v.get("score") for k, v in signals.items()}
        if req.task:
            scores["task"] = compute_task_score(req.task)

        if not any(v is not None for v in scores.values()):
            raise HTTPException(
                status_code=422,
                detail="No valid signals for baseline update."
            )

        baseline = update_baseline(req.subject_id, scores)
        return {
            "subject_id": req.subject_id,
            "baseline": baseline,
            "model_version": MODEL_VERSION
        }
    finally:
        for path in temp_files:
            remove_temp_file(path)


@app.post("/process", response_model=ScanResponse)
def process_scan(req: ScanRequest):
    signals, temp_files = _analyze_media(req.media)
    try:
        logger.info(
            "process_scan scan_id=%s subject_id=%s",
            req.scan_id,
            req.subject_id
        )

        camera_score = signals["camera"].get("score")
        video_score = signals["video"].get("score")
        voice_score = signals["voice"].get("score")

        baseline = get_baseline(req.subject_id) if req.subject_id else None

        feature_map, _ = features_from_signals(signals, task=req.task)
        feature_vector = vector_from_features(feature_map)

        ml_result = (
            ml_runtime.predict(feature_vector)
            if ml_runtime.is_loaded()
            else None
        )

        result = compute_result(
            camera_score,
            video_score,
            voice_score,
            previous_confidence=req.previous_confidence,
            task=req.task,
            baseline=baseline,
            ml_result=ml_result,
        )

        diagnostics = {
            "missing_media": result.get("missing_media", []),
            "signals": signals,
            "confidence_components": result.get("confidence_components", {}),
            "baseline": baseline,
            "ml": ml_result,
        }

        response = dict(result)
        response["diagnostics"] = diagnostics
        response["model_version"] = MODEL_VERSION

        return response
    finally:
        for path in temp_files:
            remove_temp_file(path)

