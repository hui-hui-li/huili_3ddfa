from __future__ import annotations

import base64
import json
import os
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import cv2
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.dependencies import get_current_user
from app.core.models import ReconstructionJob, User
from app.core.schemas import (
    FaceSwapFrameResponse,
    FaceSwapSourceOption,
    RealtimeAttentionFace,
)
from app.services.reconstruction import (
    analyze_attention_frame,
    extract_face_template,
    swap_face_in_image_bytes,
)


router = APIRouter(prefix="/face-swap", tags=["face-swap"])


_template_lock = threading.Lock()
_template_cache: Dict[str, Dict[str, object]] = {}
_template_ttl_seconds = 20 * 60
_track_lock = threading.Lock()
_target_track_cache: Dict[str, Dict[str, object]] = {}
_target_track_ttl_seconds = 8 * 60


def _cleanup_template_cache() -> None:
    now = time.time()
    with _template_lock:
        for key in list(_template_cache.keys()):
            entry = _template_cache.get(key) or {}
            updated = float(entry.get("updated_at", 0.0) or 0.0)
            if now - updated > _template_ttl_seconds:
                _template_cache.pop(key, None)


def _cleanup_target_track_cache() -> None:
    now = time.time()
    with _track_lock:
        for key in list(_target_track_cache.keys()):
            entry = _target_track_cache.get(key) or {}
            updated = float(entry.get("updated_at", 0.0) or 0.0)
            if now - updated > _target_track_ttl_seconds:
                _target_track_cache.pop(key, None)


def _normalize_scenario(value: object) -> str:
    key = str(value or "classroom").strip().lower()
    if key in {"classroom", "exam", "driving"}:
        return key
    return "classroom"


def _get_user_job(db: Session, current_user: User, job_id: str) -> ReconstructionJob:
    job = (
        db.query(ReconstructionJob)
        .filter(ReconstructionJob.id == job_id, ReconstructionJob.user_id == current_user.id)
        .first()
    )
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Reconstruction job not found")
    return job


def _require_completed_job(db: Session, current_user: User, job_id: str) -> ReconstructionJob:
    job = _get_user_job(db, current_user, job_id)
    if job.status != "completed":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Reconstruction job must be completed")
    media = getattr(job, "media", None)
    if media is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Source media not found")
    media_path = str(getattr(media, "stored_path", "") or "")
    if not media_path or not os.path.exists(media_path):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Source media file missing")
    return job


def _read_keyframe_index(job: ReconstructionJob) -> Optional[int]:
    metadata_path = str(job.output_metadata_path or "")
    if not metadata_path or not os.path.exists(metadata_path):
        return None
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        keyframe = payload.get("keyframe_index")
        if keyframe is None:
            return None
        keyframe_int = int(keyframe)
        return keyframe_int if keyframe_int >= 0 else None
    except Exception:
        return None


def _cache_key(
    *,
    user_id: int,
    job_id: str,
    source_face_index: int,
    keyframe_index: Optional[int],
) -> str:
    keyframe_text = "na" if keyframe_index is None else str(int(keyframe_index))
    return "{}:{}:{}:{}".format(int(user_id), str(job_id), int(source_face_index), keyframe_text)


def _track_key(
    *,
    user_id: int,
    session_id: str,
    source_job_id: str,
    target_face_index: int,
) -> str:
    return "{}:{}:{}:{}".format(int(user_id), str(session_id), str(source_job_id), int(target_face_index))


def _normalize_box(box: object) -> Optional[List[float]]:
    if not isinstance(box, (list, tuple)) or len(box) < 4:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in box[:4]]
    except Exception:
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _get_tracked_box(key: str) -> Optional[List[float]]:
    with _track_lock:
        entry = _target_track_cache.get(key)
        if not entry:
            return None
        box = _normalize_box(entry.get("target_box"))
        if box is None:
            return None
        entry["updated_at"] = time.time()
        return box


def _set_tracked_box(key: str, box: List[float]) -> None:
    with _track_lock:
        _target_track_cache[key] = {
            "target_box": list(box),
            "updated_at": time.time(),
        }


def _clear_tracked_box(key: str) -> None:
    with _track_lock:
        _target_track_cache.pop(key, None)


def _get_source_template(
    *,
    db: Session,
    current_user: User,
    source_job_id: str,
    source_face_index: int,
) -> Tuple[Dict[str, object], ReconstructionJob]:
    job = _require_completed_job(db, current_user, source_job_id)
    media = job.media
    media_type = str(media.media_type or "").strip().lower()
    if media_type not in {"photo", "video"}:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Unsupported source media type")

    keyframe_index = _read_keyframe_index(job) if media_type == "video" else None
    key = _cache_key(
        user_id=current_user.id,
        job_id=job.id,
        source_face_index=source_face_index,
        keyframe_index=keyframe_index,
    )

    _cleanup_template_cache()
    with _template_lock:
        cached = _template_cache.get(key)
        if cached and cached.get("template") is not None:
            cached["updated_at"] = time.time()
            return dict(cached["template"]), job

    try:
        template = extract_face_template(
            media_path=str(media.stored_path),
            media_type=media_type,
            source_face_index=source_face_index,
            keyframe_index=keyframe_index,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    with _template_lock:
        _template_cache[key] = {
            "template": template,
            "updated_at": time.time(),
        }
    return dict(template), job


@router.get("/sources", response_model=List[FaceSwapSourceOption])
def list_swap_sources(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> List[FaceSwapSourceOption]:
    jobs = (
        db.query(ReconstructionJob)
        .filter(
            ReconstructionJob.user_id == current_user.id,
            ReconstructionJob.status == "completed",
        )
        .order_by(ReconstructionJob.created_at.desc())
        .all()
    )

    options: List[FaceSwapSourceOption] = []
    for job in jobs:
        media = getattr(job, "media", None)
        media_type = str(getattr(media, "media_type", "") or "").strip().lower()
        media_path = str(getattr(media, "stored_path", "") or "")
        if media_type not in {"photo", "video"}:
            continue
        if not media_path or not os.path.exists(media_path):
            continue
        task_name = (job.task_name or "").strip() or "media-{}".format(job.media_id)
        options.append(
            FaceSwapSourceOption(
                job_id=str(job.id),
                task_name=task_name,
                media_type=media_type,  # type: ignore[arg-type]
                attention_scenario=_normalize_scenario(job.attention_scenario),  # type: ignore[arg-type]
                created_at=job.created_at or datetime.utcnow(),
            )
        )
    return options


@router.post("/frame", response_model=FaceSwapFrameResponse)
def swap_face_frame(
    file: UploadFile = File(...),
    source_job_id: str = Query(..., min_length=3, max_length=64),
    scenario: str = Query(default="classroom", pattern="^(classroom|exam|driving)$"),
    mode: str = Query(default="single", pattern="^(single|multi)$"),
    session_id: Optional[str] = Query(default=None, min_length=3, max_length=64),
    track_target_face: bool = Query(default=True),
    lock_x: Optional[float] = Query(default=None, ge=0.0, le=1.0),
    lock_y: Optional[float] = Query(default=None, ge=0.0, le=1.0),
    source_face_index: int = Query(default=0, ge=0, le=12),
    target_face_index: int = Query(default=0, ge=0, le=12),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> FaceSwapFrameResponse:
    image_bytes = file.file.read()
    if not image_bytes:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Empty image file")

    source_template, _ = _get_source_template(
        db=db,
        current_user=current_user,
        source_job_id=source_job_id,
        source_face_index=source_face_index,
    )

    lock_point: Optional[Tuple[float, float]] = None
    if lock_x is not None and lock_y is not None:
        lock_point = (float(lock_x), float(lock_y))

    _cleanup_target_track_cache()
    tracking_key: Optional[str] = None
    previous_target_box: Optional[List[float]] = None
    if track_target_face and session_id:
        tracking_key = _track_key(
            user_id=current_user.id,
            session_id=session_id,
            source_job_id=source_job_id,
            target_face_index=target_face_index,
        )
        previous_target_box = _get_tracked_box(tracking_key)

    try:
        swapped = swap_face_in_image_bytes(
            image_bytes=image_bytes,
            source_template=source_template,
            target_face_index=target_face_index,
            profile="realtime",
            previous_target_box=previous_target_box,
            enable_tracking=tracking_key is not None,
            lock_point=lock_point,
        )
        analysis = analyze_attention_frame(image_bytes, scenario=scenario, mode=mode)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    if tracking_key:
        current_box = _normalize_box(swapped.get("target_box"))
        if current_box is not None:
            _set_tracked_box(tracking_key, current_box)
        elif int(swapped.get("face_count", 0)) <= 0:
            _clear_tracked_box(tracking_key)

    out_frame = swapped.get("frame")
    if out_frame is None:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Swap output missing")
    ok, encoded = cv2.imencode(".jpg", out_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    if not ok:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to encode swapped frame")
    swapped_b64 = base64.b64encode(encoded.tobytes()).decode("ascii")

    faces_raw = list(analysis.get("faces", []) or [])
    face_models = [
        RealtimeAttentionFace(
            face_index=int(face.get("face_index", idx)),
            yaw=float(face.get("yaw", 0.0)),
            pitch=float(face.get("pitch", 0.0)),
            roll=float(face.get("roll", 0.0)),
            attention_score=float(face.get("attention_score", 0.0)),
            bbox_x=(float(face.get("bbox_x")) if face.get("bbox_x") is not None else None),
            bbox_y=(float(face.get("bbox_y")) if face.get("bbox_y") is not None else None),
            bbox_w=(float(face.get("bbox_w")) if face.get("bbox_w") is not None else None),
            bbox_h=(float(face.get("bbox_h")) if face.get("bbox_h") is not None else None),
            head_down=bool(face.get("head_down", False)),
            side_view=bool(face.get("side_view", False)),
            tilted=bool(face.get("tilted", False)),
            distracted=bool(face.get("distracted", False)),
        )
        for idx, face in enumerate(faces_raw)
    ]

    return FaceSwapFrameResponse(
        mode="multi" if str(analysis.get("mode") or "single") == "multi" else "single",
        scenario=_normalize_scenario(analysis.get("scenario") or scenario),  # type: ignore[arg-type]
        timestamp=datetime.utcnow(),
        source_job_id=str(source_job_id),
        selected_target_face_index=int(swapped.get("target_face_index", target_face_index)),
        face_count=int(analysis.get("face_count", len(face_models))),
        avg_attention=float(analysis.get("avg_attention", 0.0)),
        classroom_head_up_rate=float(analysis.get("classroom_head_up_rate", 0.0)),
        replaced=bool(swapped.get("replaced", False)),
        swapped_image_base64=swapped_b64,
        faces=face_models,
    )
