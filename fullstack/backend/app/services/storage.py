from __future__ import annotations
import shutil
from pathlib import Path
from typing import Tuple
from uuid import uuid4

from fastapi import HTTPException, UploadFile, status

from app.core.config import settings


def ensure_storage_dirs() -> None:
    settings.storage_root.mkdir(parents=True, exist_ok=True)
    settings.uploads_root.mkdir(parents=True, exist_ok=True)
    settings.outputs_root.mkdir(parents=True, exist_ok=True)


def _safe_extension(filename: str) -> str:
    return Path(filename).suffix.lower().strip()


def _validate_extension(media_type: str, filename: str) -> str:
    ext = _safe_extension(filename)
    allowed = settings.photo_exts if media_type == "photo" else settings.video_exts
    if ext not in allowed:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported {media_type} format: {ext}",
        )
    return ext


def save_media_upload(upload_file: UploadFile, user_id: int, media_type: str) -> Tuple[str, str, int]:
    ext = _validate_extension(media_type, upload_file.filename or "")

    user_dir = settings.uploads_root / str(user_id) / media_type
    user_dir.mkdir(parents=True, exist_ok=True)

    stored_filename = "{}{}".format(uuid4().hex, ext)
    destination = user_dir / stored_filename

    size_limit = settings.max_upload_size_mb * 1024 * 1024
    size_bytes = 0
    chunk_size = 1024 * 1024  # 1MB

    try:
        with destination.open("wb") as out_file:
            while True:
                chunk = upload_file.file.read(chunk_size)
                if not chunk:
                    break
                size_bytes += len(chunk)
                if size_bytes > size_limit:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"File is too large. Max size is {settings.max_upload_size_mb}MB.",
                    )
                out_file.write(chunk)
    except Exception:
        destination.unlink(missing_ok=True)
        raise

    return str(destination), stored_filename, size_bytes


def create_job_output_dir(user_id: int, job_id: str) -> Path:
    output_dir = settings.outputs_root / str(user_id) / job_id
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def delete_job_output_dir(user_id: int, job_id: str) -> None:
    output_dir = settings.outputs_root / str(user_id) / job_id
    if output_dir.exists():
        shutil.rmtree(output_dir, ignore_errors=True)


def delete_user_storage(user_id: int) -> None:
    uploads_dir = settings.uploads_root / str(user_id)
    outputs_dir = settings.outputs_root / str(user_id)
    if uploads_dir.exists():
        shutil.rmtree(uploads_dir, ignore_errors=True)
    if outputs_dir.exists():
        shutil.rmtree(outputs_dir, ignore_errors=True)
