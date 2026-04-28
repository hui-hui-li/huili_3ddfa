from __future__ import annotations

import asyncio
import io
import json
import os
import shutil
import tempfile
from datetime import date, datetime, time, timedelta
from pathlib import Path
from queue import Empty
from typing import List, Optional
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import FileResponse, Response, StreamingResponse
from sqlalchemy import or_
from sqlalchemy.orm import Session
from starlette.background import BackgroundTask

from app.core.database import get_db
from app.core.dependencies import get_current_user, get_current_user_for_download
from app.core.models import MediaAsset, ReconstructionJob, User
from app.core.schemas import (
    AttentionCurvePoint,
    AttentionCurveResponse,
    AttentionSummaryResponse,
    AttentionTimelineEntry,
    AttentionTimelineResponse,
    FramePreviewEntry,
    FramePreviewListResponse,
    ReconstructionBatchCancelRequest,
    ReconstructionBatchCancelResponse,
    ReconstructionBatchCreateItem,
    ReconstructionBatchCreateRequest,
    ReconstructionBatchCreateResponse,
    ReconstructionBatchDeleteRequest,
    ReconstructionBatchDeleteResponse,
    ReconstructionCreateRequest,
    ReconstructionResponse,
)
from app.services.job_queue import (
    emit_job_event,
    is_job_active,
    request_job_cancel,
    submit_job,
    subscribe_job_events,
    unsubscribe_job_events,
)
from app.services.reconstruction import (
    build_attention_summary_from_entries,
    export_video_animation,
    export_video_sequence_zip,
    score_attention_from_pose,
)
from app.services.storage import delete_job_output_dir


router = APIRouter(prefix="/reconstructions", tags=["reconstructions"])


def _job_url(job_id: str, endpoint: str, has_file: bool) -> Optional[str]:
    if not has_file:
        return None
    return "/api/reconstructions/{}/{}".format(job_id, endpoint)


def _path_exists(path: Optional[str]) -> bool:
    return bool(path) and os.path.exists(str(path))


def _can_generate_video_download(job: ReconstructionJob, media_type: str) -> bool:
    media = getattr(job, "media", None)
    stored_path = getattr(media, "stored_path", None)
    return media_type == "video" and job.status == "completed" and _path_exists(stored_path)


def _cleanup_temp_dir(temp_dir: str) -> None:
    shutil.rmtree(temp_dir, ignore_errors=True)


def _require_original_video_path(job: ReconstructionJob) -> str:
    media = getattr(job, "media", None)
    if getattr(media, "media_type", None) != "video" or job.status != "completed":
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Video output not found")

    stored_path = str(getattr(media, "stored_path", "") or "")
    if not stored_path or not os.path.exists(stored_path):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Original video not found")
    return stored_path


def _build_job_response(job: ReconstructionJob) -> ReconstructionResponse:
    task_name = (job.task_name or "").strip()
    media = getattr(job, "media", None)
    media_type = getattr(media, "media_type", None) or "photo"
    if not task_name:
        task_name = getattr(media, "original_filename", "") or "media-{}".format(job.media_id)
    return ReconstructionResponse(
        id=job.id,
        media_id=job.media_id,
        task_name=task_name,
        media_type=media_type,
        status=job.status,
        attention_scenario=str(job.attention_scenario or "classroom"),
        output_model_path=_job_url(job.id, "download", bool(job.output_model_path)),
        output_preview_path=_job_url(job.id, "preview", bool(job.output_preview_path)),
        output_sequence_zip_path=_job_url(
            job.id,
            "sequence",
            _path_exists(job.output_sequence_zip_path) or _can_generate_video_download(job, media_type),
        ),
        output_animation_path=_job_url(
            job.id,
            "animation",
            _path_exists(job.output_animation_path) or _can_generate_video_download(job, media_type),
        ),
        output_metadata_path=_job_url(job.id, "metadata", bool(job.output_metadata_path)),
        output_attention_metadata_path=_job_url(job.id, "attention-metadata", bool(job.output_attention_metadata_path)),
        progress_percent=int(job.progress_percent or 0),
        progress_stage=job.progress_stage,
        progress_message=job.progress_message,
        total_frames=job.total_frames,
        processed_frames=job.processed_frames,
        keyframe_index=job.keyframe_index,
        error_message=job.error_message,
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
    )


def _has_active_job_for_media(db: Session, user_id: int, media_id: int) -> bool:
    active = (
        db.query(ReconstructionJob.id)
        .filter(
            ReconstructionJob.user_id == user_id,
            ReconstructionJob.media_id == media_id,
            ReconstructionJob.status.in_(("queued", "running")),
        )
        .first()
    )
    return active is not None


def _normalize_attention_scenario(value: object) -> Optional[str]:
    normalized = str(value or "").strip().lower()
    if normalized in {"classroom", "exam", "driving"}:
        return normalized
    return None


def _infer_attention_scenario_from_filename(filename: object) -> Optional[str]:
    text = str(filename or "").strip().lower()
    if not text:
        return None
    if "驾驶" in text or "driving" in text:
        return "driving"
    if "考试" in text or "exam" in text:
        return "exam"
    if "课堂" in text or "上课" in text or "classroom" in text:
        return "classroom"
    return None


def _resolve_media_default_attention_scenario(db: Session, media: MediaAsset) -> str:
    media_value = _normalize_attention_scenario(getattr(media, "default_attention_scenario", None))
    if media_value is not None:
        return media_value

    scenario = (
        db.query(ReconstructionJob.attention_scenario)
        .filter(ReconstructionJob.media_id == media.id)
        .order_by(ReconstructionJob.created_at.asc(), ReconstructionJob.id.asc())
        .scalar()
    )
    return (
        _normalize_attention_scenario(scenario)
        or _infer_attention_scenario_from_filename(getattr(media, "original_filename", None))
        or "classroom"
    )


def _build_new_job(media: MediaAsset, user_id: int, attention_scenario: str) -> ReconstructionJob:
    return ReconstructionJob(
        id=str(uuid4()),
        user_id=user_id,
        media_id=media.id,
        task_name=(media.original_filename or "media-{}".format(media.id)),
        attention_scenario=attention_scenario,
        status="queued",
        progress_percent=0,
        progress_stage="queued",
        progress_message="job queued",
        created_at=datetime.utcnow(),
    )


@router.post("", response_model=ReconstructionResponse)
def create_reconstruction(
    payload: ReconstructionCreateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> ReconstructionResponse:
    media = (
        db.query(MediaAsset)
        .filter(MediaAsset.id == payload.media_id, MediaAsset.user_id == current_user.id)
        .first()
    )
    if media is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Media not found")
    if _has_active_job_for_media(db, current_user.id, media.id):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Media already has an active reconstruction job")

    attention_scenario = payload.attention_scenario or _resolve_media_default_attention_scenario(db, media)
    job = _build_new_job(media, current_user.id, attention_scenario)
    db.add(job)
    db.commit()
    db.refresh(job)

    submit_job(job.id)
    return _build_job_response(job)


@router.get("", response_model=List[ReconstructionResponse])
def list_reconstructions(
    media_id: Optional[int] = Query(default=None),
    status_value: Optional[str] = Query(default=None, alias="status"),
    attention_scenario: Optional[str] = Query(default=None, pattern="^(classroom|exam|driving)$"),
    search: Optional[str] = Query(default=None),
    created_from: Optional[date] = Query(default=None),
    created_to: Optional[date] = Query(default=None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> List[ReconstructionResponse]:
    query = db.query(ReconstructionJob).filter(ReconstructionJob.user_id == current_user.id)
    if media_id is not None:
        query = query.filter(ReconstructionJob.media_id == media_id)
    if status_value is not None:
        query = query.filter(ReconstructionJob.status == status_value)
    if attention_scenario is not None:
        query = query.filter(ReconstructionJob.attention_scenario == attention_scenario)
    if search is not None:
        keyword = search.strip()
        if keyword:
            pattern = "%{}%".format(keyword)
            query = query.filter(
                or_(
                    ReconstructionJob.task_name.like(pattern),
                    ReconstructionJob.id.like(pattern),
                )
            )
    if created_from is not None:
        query = query.filter(ReconstructionJob.created_at >= datetime.combine(created_from, time.min))
    if created_to is not None:
        query = query.filter(ReconstructionJob.created_at < datetime.combine(created_to + timedelta(days=1), time.min))

    jobs = query.order_by(ReconstructionJob.created_at.desc()).all()
    return [_build_job_response(job) for job in jobs]


@router.get("/{job_id}", response_model=ReconstructionResponse)
def get_reconstruction(
    job_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> ReconstructionResponse:
    job = (
        db.query(ReconstructionJob)
        .filter(ReconstructionJob.id == job_id, ReconstructionJob.user_id == current_user.id)
        .first()
    )
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Reconstruction job not found")
    return _build_job_response(job)


@router.delete("/{job_id}")
def delete_reconstruction(
    job_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    job = _get_user_job(db, current_user, job_id)
    if job.status in {"queued", "running"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete reconstruction while it is queued or running",
        )

    delete_job_output_dir(job.user_id, job.id)
    db.delete(job)
    db.commit()
    return {"detail": "deleted"}


@router.post("/batch/create", response_model=ReconstructionBatchCreateResponse)
def batch_create_reconstructions(
    payload: ReconstructionBatchCreateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> ReconstructionBatchCreateResponse:
    media_ids: List[int] = []
    for raw_id in payload.media_ids:
        media_id = int(raw_id)
        if media_id not in media_ids:
            media_ids.append(media_id)

    if not media_ids:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No media ids provided")

    media_items = (
        db.query(MediaAsset)
        .filter(MediaAsset.user_id == current_user.id, MediaAsset.id.in_(media_ids))
        .all()
    )
    media_map = {item.id: item for item in media_items}
    active_media_ids = {
        row[0]
        for row in db.query(ReconstructionJob.media_id)
        .filter(
            ReconstructionJob.user_id == current_user.id,
            ReconstructionJob.media_id.in_(media_ids),
            ReconstructionJob.status.in_(("queued", "running")),
        )
        .all()
    }

    created: List[ReconstructionBatchCreateItem] = []
    blocked_media_ids: List[int] = []
    missing_media_ids: List[int] = []
    created_job_ids: List[str] = []

    for media_id in media_ids:
        media = media_map.get(media_id)
        if media is None:
            missing_media_ids.append(media_id)
            continue
        if media_id in active_media_ids:
            blocked_media_ids.append(media_id)
            continue

        attention_scenario = payload.attention_scenario or _resolve_media_default_attention_scenario(db, media)
        job = _build_new_job(media, current_user.id, attention_scenario)
        db.add(job)
        created_job_ids.append(job.id)
        created.append(
            ReconstructionBatchCreateItem(
                media_id=media_id,
                job_id=job.id,
                attention_scenario=attention_scenario,
            )
        )

    db.commit()
    for job_id in created_job_ids:
        submit_job(job_id)

    detail = "created={} blocked={} missing={}".format(
        len(created),
        len(blocked_media_ids),
        len(missing_media_ids),
    )
    return ReconstructionBatchCreateResponse(
        created=created,
        blocked_media_ids=blocked_media_ids,
        missing_media_ids=missing_media_ids,
        detail=detail,
    )


@router.post("/batch/delete", response_model=ReconstructionBatchDeleteResponse)
def batch_delete_reconstructions(
    payload: ReconstructionBatchDeleteRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> ReconstructionBatchDeleteResponse:
    ids: List[str] = []
    for raw_id in payload.job_ids:
        job_id = str(raw_id).strip()
        if job_id and job_id not in ids:
            ids.append(job_id)

    if not ids:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No job ids provided")

    jobs = (
        db.query(ReconstructionJob)
        .filter(ReconstructionJob.user_id == current_user.id, ReconstructionJob.id.in_(ids))
        .all()
    )
    job_map = {job.id: job for job in jobs}

    deleted_ids: List[str] = []
    blocked_ids: List[str] = []
    missing_ids: List[str] = []

    for job_id in ids:
        job = job_map.get(job_id)
        if job is None:
            missing_ids.append(job_id)
            continue
        if job.status in {"queued", "running"}:
            blocked_ids.append(job_id)
            continue
        delete_job_output_dir(job.user_id, job.id)
        db.delete(job)
        deleted_ids.append(job_id)

    db.commit()

    detail = "deleted={} blocked={} missing={}".format(
        len(deleted_ids),
        len(blocked_ids),
        len(missing_ids),
    )
    return ReconstructionBatchDeleteResponse(
        deleted_ids=deleted_ids,
        blocked_ids=blocked_ids,
        missing_ids=missing_ids,
        detail=detail,
    )


def _cancel_reconstruction_job(db: Session, job: ReconstructionJob) -> ReconstructionJob:
    if job.status in {"completed", "failed", "cancelled"}:
        return job

    request_job_cancel(job.id)
    if job.status == "queued":
        job.status = "cancelled"
        job.progress_stage = "cancelled"
        job.progress_message = "cancelled by user before start"
        job.progress_percent = 100
        job.error_message = None
        job.finished_at = datetime.utcnow()
        db.commit()
        db.refresh(job)
        emit_job_event(job, event_type="cancelled")
        return job

    if not is_job_active(job.id):
        job.status = "cancelled"
        job.progress_stage = "cancelled"
        job.progress_message = "cancelled after worker became unavailable"
        job.progress_percent = 100
        job.error_message = None
        job.finished_at = datetime.utcnow()
        db.commit()
        db.refresh(job)
        emit_job_event(job, event_type="cancelled")
        return job

    job.progress_stage = "cancel_requested"
    job.progress_message = "cancel requested by user"
    if job.progress_percent < 1:
        job.progress_percent = 1
    db.commit()
    db.refresh(job)
    emit_job_event(job, event_type="progress")
    return job


@router.post("/batch/cancel", response_model=ReconstructionBatchCancelResponse)
def batch_cancel_reconstructions(
    payload: ReconstructionBatchCancelRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> ReconstructionBatchCancelResponse:
    ids: List[str] = []
    for raw_id in payload.job_ids:
        job_id = str(raw_id).strip()
        if job_id and job_id not in ids:
            ids.append(job_id)

    if not ids:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No job ids provided")

    jobs = (
        db.query(ReconstructionJob)
        .filter(ReconstructionJob.user_id == current_user.id, ReconstructionJob.id.in_(ids))
        .all()
    )
    job_map = {job.id: job for job in jobs}

    requested_ids: List[str] = []
    cancelled_ids: List[str] = []
    already_terminal_ids: List[str] = []
    missing_ids: List[str] = []

    for job_id in ids:
        job = job_map.get(job_id)
        if job is None:
            missing_ids.append(job_id)
            continue
        if job.status in {"completed", "failed", "cancelled"}:
            already_terminal_ids.append(job_id)
            continue
        _cancel_reconstruction_job(db, job)
        if job.status == "cancelled":
            cancelled_ids.append(job_id)
        else:
            requested_ids.append(job_id)

    detail = "requested={} cancelled={} terminal={} missing={}".format(
        len(requested_ids),
        len(cancelled_ids),
        len(already_terminal_ids),
        len(missing_ids),
    )
    return ReconstructionBatchCancelResponse(
        requested_ids=requested_ids,
        cancelled_ids=cancelled_ids,
        already_terminal_ids=already_terminal_ids,
        missing_ids=missing_ids,
        detail=detail,
    )


@router.post("/{job_id}/cancel", response_model=ReconstructionResponse)
def cancel_reconstruction(
    job_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> ReconstructionResponse:
    job = _get_user_job(db, current_user, job_id)
    _cancel_reconstruction_job(db, job)
    return _build_job_response(job)


@router.get("/{job_id}/events")
async def stream_reconstruction_events(
    job_id: str,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user_for_download),
):
    job = _get_user_job(db, current_user, job_id)
    subscriber = subscribe_job_events(job_id)
    terminal_status = {"completed", "failed", "cancelled"}

    async def event_generator():
        try:
            snapshot = json.loads(_build_job_response(job).json())
            snapshot["event"] = "snapshot"
            snapshot["job_id"] = job.id
            yield "event: snapshot\ndata: {}\n\n".format(json.dumps(snapshot, ensure_ascii=False))

            if job.status in terminal_status:
                return

            while True:
                if await request.is_disconnected():
                    return
                try:
                    loop = asyncio.get_running_loop()
                    payload = await loop.run_in_executor(None, subscriber.get, True, 1.0)
                except Empty:
                    yield ": ping\n\n"
                    continue
                event_type = str(payload.get("event") or "progress")
                yield "event: {}\ndata: {}\n\n".format(event_type, json.dumps(payload, ensure_ascii=False))
                if payload.get("status") in terminal_status:
                    return
        finally:
            unsubscribe_job_events(job_id, subscriber)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def _get_user_job(db: Session, current_user: User, job_id: str) -> ReconstructionJob:
    job = (
        db.query(ReconstructionJob)
        .filter(ReconstructionJob.id == job_id, ReconstructionJob.user_id == current_user.id)
        .first()
    )
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Reconstruction job not found")
    return job


def _load_metadata(job: ReconstructionJob) -> dict:
    if not job.output_metadata_path or not os.path.exists(job.output_metadata_path):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Metadata output not found")
    with open(job.output_metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_attention_metadata(job: ReconstructionJob) -> dict:
    if not job.output_attention_metadata_path or not os.path.exists(job.output_attention_metadata_path):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Attention metadata output not found")
    with open(job.output_attention_metadata_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return _normalize_attention_metadata(payload, str(job.attention_scenario or "classroom"))


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _safe_bool(value: object) -> bool:
    return bool(value)


def _moving_avg(entries: List[dict], index: int, window: int = 9) -> float:
    if not entries:
        return 0.0
    half = max(0, window // 2)
    lo = max(0, index - half)
    hi = min(len(entries), index + half + 1)
    if hi <= lo:
        return _safe_float(entries[index].get("attention_score", 0.0))
    total = 0.0
    count = 0
    for idx in range(lo, hi):
        total += _safe_float(entries[idx].get("attention_score", 0.0))
        count += 1
    if count <= 0:
        return 0.0
    return total / float(count)


def _normalize_attention_metadata(payload: dict, scenario: str) -> dict:
    scenario_key = str(payload.get("scenario") or scenario or "classroom").strip().lower() or "classroom"
    fps = _safe_float(payload.get("fps", 25.0), 25.0)
    raw_entries = payload.get("entries", []) or []

    normalized_entries: List[dict] = []
    for item in raw_entries:
        yaw = _safe_float(item.get("yaw", 0.0))
        pitch = _safe_float(item.get("pitch", 0.0))
        roll = _safe_float(item.get("roll", 0.0))
        score, flags = score_attention_from_pose(yaw, pitch, roll, scenario_key)
        normalized_entries.append(
            {
                "frame_index": int(item.get("frame_index", 0)),
                "yaw": round(yaw, 4),
                "pitch": round(pitch, 4),
                "roll": round(roll, 4),
                "attention_score": round(float(score), 4),
                "source": str(item.get("source") or "unknown"),
                "detection_score": round(_safe_float(item.get("detection_score", 0.0)), 4),
                "head_down": bool(flags["head_down"]),
                "side_view": bool(flags["side_view"]),
                "tilted": bool(flags["tilted"]),
                "distracted": bool(flags["distracted"]),
                "rapid_turn": _safe_bool(item.get("rapid_turn")),
            }
        )

    summary_raw = payload.get("summary") or {}
    detected_frames = int(summary_raw.get("detected_frames", 0))
    interpolated_frames = int(summary_raw.get("interpolated_frames", 0))
    if detected_frames <= 0 and normalized_entries:
        detected_frames = sum(1 for item in normalized_entries if item.get("source") == "detected")
    if interpolated_frames <= 0 and normalized_entries:
        interpolated_frames = sum(1 for item in normalized_entries if item.get("source") != "detected")
    rapid_turn_events = sum(1 for item in normalized_entries if item.get("rapid_turn"))

    normalized_summary = build_attention_summary_from_entries(
        normalized_entries,
        fps=fps,
        detected_frames=detected_frames,
        interpolated_frames=interpolated_frames,
        scenario=scenario_key,
        rapid_turn_events=rapid_turn_events,
    )

    return {
        **payload,
        "scenario": scenario_key,
        "fps": round(float(fps), 4),
        "summary": normalized_summary,
        "entries": normalized_entries,
    }


@router.get("/{job_id}/download")
def download_model(
    job_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user_for_download),
):
    job = _get_user_job(db, current_user, job_id)
    if not job.output_model_path or not os.path.exists(job.output_model_path):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model output not found")

    return FileResponse(
        path=job.output_model_path,
        filename="{}_keyframe.obj".format(job.id),
        media_type="application/octet-stream",
    )


@router.get("/{job_id}/preview")
def download_preview(
    job_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user_for_download),
):
    job = _get_user_job(db, current_user, job_id)
    if not job.output_preview_path or not os.path.exists(job.output_preview_path):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Preview output not found")

    return FileResponse(
        path=job.output_preview_path,
        filename="{}_keyframe.jpg".format(job.id),
        media_type="image/jpeg",
    )


@router.get("/{job_id}/sequence")
def download_sequence_zip(
    job_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user_for_download),
):
    job = _get_user_job(db, current_user, job_id)
    if job.output_sequence_zip_path and os.path.exists(job.output_sequence_zip_path):
        return FileResponse(
            path=job.output_sequence_zip_path,
            filename="{}_sequence_obj.zip".format(job.id),
            media_type="application/zip",
        )

    media_path = _require_original_video_path(job)
    temp_dir = tempfile.mkdtemp(prefix="reconstruction-sequence-")
    zip_path = Path(temp_dir) / "{}_sequence_obj.zip".format(job.id)
    try:
        export_video_sequence_zip(media_path, zip_path)
    except Exception:
        _cleanup_temp_dir(temp_dir)
        raise
    return FileResponse(
        path=str(zip_path),
        filename="{}_sequence_obj.zip".format(job.id),
        media_type="application/zip",
        background=BackgroundTask(_cleanup_temp_dir, temp_dir),
    )


@router.get("/{job_id}/animation")
def download_animation(
    job_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user_for_download),
):
    job = _get_user_job(db, current_user, job_id)
    if job.output_animation_path and os.path.exists(job.output_animation_path):
        return FileResponse(
            path=job.output_animation_path,
            filename="{}_animation.mp4".format(job.id),
            media_type="video/mp4",
        )

    media_path = _require_original_video_path(job)
    temp_dir = tempfile.mkdtemp(prefix="reconstruction-animation-")
    animation_path = Path(temp_dir) / "{}_animation.mp4".format(job.id)
    try:
        export_video_animation(media_path, animation_path)
    except Exception:
        _cleanup_temp_dir(temp_dir)
        raise
    return FileResponse(
        path=str(animation_path),
        filename="{}_animation.mp4".format(job.id),
        media_type="video/mp4",
        background=BackgroundTask(_cleanup_temp_dir, temp_dir),
    )


@router.get("/{job_id}/metadata")
def download_metadata(
    job_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user_for_download),
):
    job = _get_user_job(db, current_user, job_id)
    if not job.output_metadata_path or not os.path.exists(job.output_metadata_path):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Metadata output not found")

    return FileResponse(
        path=job.output_metadata_path,
        filename="{}_metadata.json".format(job.id),
        media_type="application/json",
    )


@router.get("/{job_id}/attention-metadata")
def download_attention_metadata(
    job_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user_for_download),
):
    job = _get_user_job(db, current_user, job_id)
    attention_meta = _load_attention_metadata(job)
    return Response(
        content=json.dumps(attention_meta, ensure_ascii=False, indent=2),
        media_type="application/json",
        headers={
            "Content-Disposition": 'attachment; filename="{}_attention_metadata.json"'.format(job.id),
        },
    )


@router.get("/{job_id}/attention", response_model=AttentionSummaryResponse)
def get_attention_summary(
    job_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> AttentionSummaryResponse:
    job = _get_user_job(db, current_user, job_id)
    attention_meta = _load_attention_metadata(job)
    summary = attention_meta.get("summary") or {}
    return AttentionSummaryResponse(
        scenario=str(summary.get("scenario", "classroom")),
        fps=float(summary.get("fps", attention_meta.get("fps", 25.0))),
        total_frames=int(summary.get("total_frames", 0)),
        detected_frames=int(summary.get("detected_frames", 0)),
        interpolated_frames=int(summary.get("interpolated_frames", 0)),
        avg_attention=float(summary.get("avg_attention", 0.0)),
        min_attention=float(summary.get("min_attention", 0.0)),
        max_attention=float(summary.get("max_attention", 0.0)),
        low_attention_ratio=float(summary.get("low_attention_ratio", 0.0)),
        head_down_ratio=float(summary.get("head_down_ratio", 0.0)),
        side_view_ratio=float(summary.get("side_view_ratio", 0.0)),
        rapid_turn_events=int(summary.get("rapid_turn_events", 0)),
        longest_distracted_frames=int(summary.get("longest_distracted_frames", 0)),
        classroom_head_up_rate=float(summary.get("classroom_head_up_rate", 0.0)),
        exam_focus_score=float(summary.get("exam_focus_score", 0.0)),
        driving_risk_score=float(summary.get("driving_risk_score", 0.0)),
        warnings=[str(v) for v in (summary.get("warnings", []) or [])],
    )


@router.get("/{job_id}/attention-timeline", response_model=AttentionTimelineResponse)
def list_attention_timeline(
    job_id: str,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=60, ge=1, le=500),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> AttentionTimelineResponse:
    job = _get_user_job(db, current_user, job_id)
    attention_meta = _load_attention_metadata(job)
    raw_entries = attention_meta.get("entries", []) or []

    entries: List[AttentionTimelineEntry] = []
    for item in raw_entries:
        if item.get("frame_index") is None:
            continue
        entries.append(
            AttentionTimelineEntry(
                frame_index=int(item.get("frame_index", 0)),
                yaw=_safe_float(item.get("yaw", 0.0)),
                pitch=_safe_float(item.get("pitch", 0.0)),
                roll=_safe_float(item.get("roll", 0.0)),
                attention_score=_safe_float(item.get("attention_score", 0.0)),
                source=str(item.get("source") or "unknown"),
                detection_score=_safe_float(item.get("detection_score", 0.0)),
                head_down=_safe_bool(item.get("head_down")),
                side_view=_safe_bool(item.get("side_view")),
                tilted=_safe_bool(item.get("tilted")),
                distracted=_safe_bool(item.get("distracted")),
                rapid_turn=_safe_bool(item.get("rapid_turn")),
            )
        )

    total = len(entries)
    start = (page - 1) * page_size
    end = start + page_size
    return AttentionTimelineResponse(total=total, page=page, page_size=page_size, entries=entries[start:end])


@router.get("/{job_id}/attention-csv")
def download_attention_csv(
    job_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user_for_download),
):
    job = _get_user_job(db, current_user, job_id)
    attention_meta = _load_attention_metadata(job)
    entries = attention_meta.get("entries", []) or []

    columns = [
        "frame_index",
        "yaw",
        "pitch",
        "roll",
        "attention_score",
        "source",
        "detection_score",
        "head_down",
        "side_view",
        "tilted",
        "distracted",
        "rapid_turn",
    ]

    output = io.StringIO()
    output.write(",".join(columns) + "\n")
    for entry in entries:
        values = []
        for col in columns:
            value = entry.get(col, "")
            text = str(value).replace('"', '""')
            values.append('"{}"'.format(text))
        output.write(",".join(values) + "\n")

    filename = "{}_attention_timeline.csv".format(job.id)
    return Response(
        content=output.getvalue(),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="{}"'.format(filename)},
    )


@router.get("/{job_id}/attention-curve", response_model=AttentionCurveResponse)
def get_attention_curve(
    job_id: str,
    max_points: int = Query(default=260, ge=20, le=2000),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> AttentionCurveResponse:
    job = _get_user_job(db, current_user, job_id)
    attention_meta = _load_attention_metadata(job)
    raw_entries = attention_meta.get("entries", []) or []

    entries: List[dict] = []
    for item in raw_entries:
        if item.get("frame_index") is None:
            continue
        entries.append(
            {
                "frame_index": int(item.get("frame_index", 0)),
                "attention_score": _safe_float(item.get("attention_score", 0.0)),
                "distracted": _safe_bool(item.get("distracted")),
                "rapid_turn": _safe_bool(item.get("rapid_turn")),
            }
        )

    total = len(entries)
    if total == 0:
        return AttentionCurveResponse(
            scenario=str(attention_meta.get("scenario", "classroom")),
            total_frames=0,
            sampled_points=0,
            points=[],
        )

    step = 1
    if total > max_points:
        step = max(1, (total + max_points - 1) // max_points)

    sampled: List[AttentionCurvePoint] = []
    for idx in range(0, total, step):
        item = entries[idx]
        sampled.append(
            AttentionCurvePoint(
                frame_index=int(item["frame_index"]),
                attention_score=round(float(item["attention_score"]), 4),
                moving_avg_score=round(float(_moving_avg(entries, idx, window=9)), 4),
                distracted=bool(item["distracted"]),
                rapid_turn=bool(item["rapid_turn"]),
            )
        )

    return AttentionCurveResponse(
        scenario=str(attention_meta.get("scenario", "classroom")),
        total_frames=total,
        sampled_points=len(sampled),
        points=sampled,
    )


@router.get("/{job_id}/frames", response_model=FramePreviewListResponse)
def list_frame_previews(
    job_id: str,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=200),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> FramePreviewListResponse:
    job = _get_user_job(db, current_user, job_id)
    metadata = _load_metadata(job)
    entries_raw = metadata.get("entries", [])

    valid_entries = []
    output_dir = os.path.dirname(job.output_metadata_path)
    for entry in entries_raw:
        preview_file = entry.get("preview_file")
        frame_index = entry.get("frame_index")
        if preview_file is None or frame_index is None:
            continue
        preview_path = os.path.join(output_dir, metadata.get("preview_dir", "sequence_preview"), preview_file)
        if not os.path.exists(preview_path):
            continue
        valid_entries.append(
            FramePreviewEntry(
                frame_index=int(frame_index),
                image_url="/api/reconstructions/{}/frames/{}".format(job.id, int(frame_index)),
            )
        )

    total = len(valid_entries)
    start = (page - 1) * page_size
    end = start + page_size

    return FramePreviewListResponse(
        total=total,
        page=page,
        page_size=page_size,
        entries=valid_entries[start:end],
    )


@router.get("/{job_id}/frames/{frame_index}")
def get_frame_preview(
    job_id: str,
    frame_index: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user_for_download),
):
    job = _get_user_job(db, current_user, job_id)
    metadata = _load_metadata(job)
    entries_raw = metadata.get("entries", [])
    preview_dir = metadata.get("preview_dir", "sequence_preview")
    output_dir = os.path.dirname(job.output_metadata_path)

    preview_file = None
    for entry in entries_raw:
        entry_idx = entry.get("frame_index")
        if entry_idx is None:
            continue
        if int(entry_idx) == frame_index:
            preview_file = entry.get("preview_file")
            break

    if preview_file is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Frame preview not found")

    preview_path = os.path.join(output_dir, preview_dir, preview_file)
    if not os.path.exists(preview_path):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Frame preview file missing")

    return FileResponse(
        path=preview_path,
        filename="{}_frame_{}.jpg".format(job.id, frame_index),
        media_type="image/jpeg",
    )
