from __future__ import annotations

import os
from typing import List, Optional
from uuid import uuid4

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.dependencies import get_current_user, get_current_user_for_download
from app.core.models import MediaAsset, ReconstructionJob, User
from app.core.schemas import MediaBatchDeleteRequest, MediaBatchDeleteResponse, MediaResponse
from app.services.job_queue import submit_job
from app.services.storage import delete_job_output_dir, save_media_upload


router = APIRouter(prefix="/media", tags=["media"])


def _build_media_response(item: MediaAsset) -> MediaResponse:
    return MediaResponse(
        id=item.id,
        media_type=item.media_type,
        original_filename=item.original_filename,
        file_size=item.file_size,
        mime_type=item.mime_type,
        default_attention_scenario=str(item.default_attention_scenario or "classroom"),
        created_at=item.created_at,
    )


def _has_active_job(db: Session, media_id: int) -> bool:
    active = (
        db.query(ReconstructionJob.id)
        .filter(
            ReconstructionJob.media_id == media_id,
            ReconstructionJob.status.in_(("queued", "running")),
        )
        .first()
    )
    return active is not None


def _delete_media_and_outputs(db: Session, item: MediaAsset) -> None:
    job_ids = [
        row[0]
        for row in db.query(ReconstructionJob.id)
        .filter(ReconstructionJob.media_id == item.id)
        .all()
    ]

    if os.path.exists(item.stored_path):
        os.remove(item.stored_path)

    for job_id in job_ids:
        delete_job_output_dir(item.user_id, str(job_id))

    db.delete(item)


def _create_media_record(
    db: Session,
    current_user: User,
    media_type: str,
    upload_file: UploadFile,
    auto_reconstruct: bool,
    attention_scenario: str,
) -> MediaAsset:
    stored_path, stored_filename, file_size = save_media_upload(upload_file, current_user.id, media_type)

    media = MediaAsset(
        user_id=current_user.id,
        media_type=media_type,
        original_filename=upload_file.filename or stored_filename,
        stored_filename=stored_filename,
        stored_path=stored_path,
        mime_type=upload_file.content_type,
        file_size=file_size,
        default_attention_scenario=attention_scenario,
    )
    db.add(media)
    db.commit()
    db.refresh(media)

    if auto_reconstruct:
        job = ReconstructionJob(
            id=str(uuid4()),
            user_id=current_user.id,
            media_id=media.id,
            task_name=(media.original_filename or "media-{}".format(media.id)),
            attention_scenario=attention_scenario,
            status="queued",
            progress_percent=0,
            progress_stage="queued",
            progress_message="job queued",
        )
        db.add(job)
        db.commit()
        submit_job(job.id)

    return media


@router.post("/photos", response_model=MediaResponse)
def upload_photo(
    file: UploadFile = File(...),
    auto_reconstruct: bool = Query(default=True),
    attention_scenario: str = Query(default="classroom", pattern="^(classroom|exam|driving)$"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> MediaResponse:
    media = _create_media_record(db, current_user, "photo", file, auto_reconstruct, attention_scenario)
    return _build_media_response(media)


@router.post("/videos", response_model=MediaResponse)
def upload_video(
    file: UploadFile = File(...),
    auto_reconstruct: bool = Query(default=True),
    attention_scenario: str = Query(default="classroom", pattern="^(classroom|exam|driving)$"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> MediaResponse:
    media = _create_media_record(db, current_user, "video", file, auto_reconstruct, attention_scenario)
    return _build_media_response(media)


@router.get("", response_model=List[MediaResponse])
def list_media(
    media_type: Optional[str] = Query(default=None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> List[MediaResponse]:
    query = db.query(MediaAsset).filter(MediaAsset.user_id == current_user.id)
    if media_type in ("photo", "video"):
        query = query.filter(MediaAsset.media_type == media_type)

    items = query.order_by(MediaAsset.created_at.desc()).all()
    return [_build_media_response(item) for item in items]


@router.get("/{media_id}", response_model=MediaResponse)
def get_media(
    media_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> MediaResponse:
    item = db.query(MediaAsset).filter(MediaAsset.id == media_id, MediaAsset.user_id == current_user.id).first()
    if item is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Media not found")
    return _build_media_response(item)


@router.post("/batch/delete", response_model=MediaBatchDeleteResponse)
def batch_delete_media(
    payload: MediaBatchDeleteRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> MediaBatchDeleteResponse:
    ids = []
    for raw_id in payload.media_ids:
        media_id = int(raw_id)
        if media_id not in ids:
            ids.append(media_id)

    if not ids:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No media ids provided")

    items = (
        db.query(MediaAsset)
        .filter(MediaAsset.user_id == current_user.id, MediaAsset.id.in_(ids))
        .all()
    )
    item_map = {item.id: item for item in items}

    deleted_ids: List[int] = []
    blocked_ids: List[int] = []
    missing_ids: List[int] = []

    for media_id in ids:
        item = item_map.get(media_id)
        if item is None:
            missing_ids.append(media_id)
            continue
        if _has_active_job(db, media_id):
            blocked_ids.append(media_id)
            continue
        _delete_media_and_outputs(db, item)
        deleted_ids.append(media_id)

    db.commit()

    detail = "deleted={} blocked={} missing={}".format(
        len(deleted_ids),
        len(blocked_ids),
        len(missing_ids),
    )
    return MediaBatchDeleteResponse(
        deleted_ids=deleted_ids,
        blocked_ids=blocked_ids,
        missing_ids=missing_ids,
        detail=detail,
    )


@router.get("/{media_id}/download")
def download_media(
    media_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user_for_download),
):
    item = db.query(MediaAsset).filter(MediaAsset.id == media_id, MediaAsset.user_id == current_user.id).first()
    if item is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Media not found")

    if not os.path.exists(item.stored_path):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Media file not found on disk")

    return FileResponse(
        path=item.stored_path,
        filename=item.original_filename,
        media_type=item.mime_type or "application/octet-stream",
    )


@router.delete("/{media_id}")
def delete_media(
    media_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    item = db.query(MediaAsset).filter(MediaAsset.id == media_id, MediaAsset.user_id == current_user.id).first()
    if item is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Media not found")
    if _has_active_job(db, media_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete media while reconstruction is queued or running",
        )

    _delete_media_and_outputs(db, item)
    db.commit()
    return {"detail": "deleted"}
