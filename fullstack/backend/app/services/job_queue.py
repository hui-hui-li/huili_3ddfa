from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime
from queue import Empty, Full, Queue
import threading
import time
from typing import Any, Dict, Optional

from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.database import SessionLocal
from app.core.models import MediaAsset, ReconstructionJob
from app.services.reconstruction import JobCancelledError, run_reconstruction
from app.services.storage import create_job_output_dir


_executor = ThreadPoolExecutor(
    max_workers=settings.reconstruction_max_workers,
    thread_name_prefix="recon-worker",
)
_events_lock = threading.Lock()
_job_subscribers: dict[str, list[Queue]] = {}
_cancel_lock = threading.Lock()
_cancel_requests: set[str] = set()
_futures_lock = threading.Lock()
_job_futures: dict[str, Future] = {}


def _safe_int(value: Any, fallback: int) -> int:
    try:
        return int(value)
    except Exception:
        return fallback


def _clamp_percent(value: Any, fallback: int = 0) -> int:
    val = _safe_int(value, fallback)
    if val < 0:
        return 0
    if val > 100:
        return 100
    return val


def _event_payload(job: ReconstructionJob, event_type: str) -> Dict[str, Any]:
    return {
        "event": event_type,
        "job_id": job.id,
        "status": job.status,
        "attention_scenario": str(job.attention_scenario or "classroom"),
        "progress_percent": _clamp_percent(job.progress_percent, 0),
        "progress_stage": job.progress_stage,
        "progress_message": job.progress_message,
        "total_frames": job.total_frames,
        "processed_frames": job.processed_frames,
        "error_message": job.error_message,
        "updated_at": datetime.utcnow().isoformat(),
    }


def _publish_job_event(job: ReconstructionJob, event_type: str = "progress") -> None:
    payload = _event_payload(job, event_type)
    with _events_lock:
        queues = list(_job_subscribers.get(job.id, []))
    for queue in queues:
        try:
            queue.put_nowait(payload)
        except Full:
            try:
                queue.get_nowait()
            except Empty:
                pass
            try:
                queue.put_nowait(payload)
            except Exception:
                pass


def emit_job_event(job: ReconstructionJob, event_type: str = "progress") -> None:
    _publish_job_event(job, event_type=event_type)


def request_job_cancel(job_id: str) -> None:
    with _cancel_lock:
        _cancel_requests.add(job_id)


def _clear_cancel_request(job_id: str) -> None:
    with _cancel_lock:
        if job_id in _cancel_requests:
            _cancel_requests.remove(job_id)


def _has_cancel_request(job_id: str) -> bool:
    with _cancel_lock:
        return job_id in _cancel_requests


def _register_job_future(job_id: str, future: Future) -> None:
    with _futures_lock:
        _job_futures[job_id] = future

    def _cleanup(_future: Future) -> None:
        with _futures_lock:
            current = _job_futures.get(job_id)
            if current is _future:
                _job_futures.pop(job_id, None)

    future.add_done_callback(_cleanup)


def is_job_active(job_id: str) -> bool:
    with _futures_lock:
        future = _job_futures.get(job_id)
    if future is None:
        return False
    return not future.done()


def subscribe_job_events(job_id: str) -> Queue:
    queue: Queue = Queue(maxsize=300)
    with _events_lock:
        _job_subscribers.setdefault(job_id, []).append(queue)
    return queue


def unsubscribe_job_events(job_id: str, queue: Queue) -> None:
    with _events_lock:
        subscribers = _job_subscribers.get(job_id, [])
        if queue in subscribers:
            subscribers.remove(queue)
        if not subscribers and job_id in _job_subscribers:
            _job_subscribers.pop(job_id, None)


def _update_job_progress(
    db: Session,
    job: ReconstructionJob,
    *,
    percent: Optional[int] = None,
    stage: Optional[str] = None,
    message: Optional[str] = None,
    total_frames: Optional[int] = None,
    processed_frames: Optional[int] = None,
    event_type: str = "progress",
    commit: bool = True,
) -> None:
    if percent is not None:
        job.progress_percent = _clamp_percent(percent, _clamp_percent(job.progress_percent, 0))
    if stage is not None:
        job.progress_stage = stage
    if message is not None:
        job.progress_message = message
    if total_frames is not None:
        job.total_frames = _safe_int(total_frames, 0)
    if processed_frames is not None:
        job.processed_frames = _safe_int(processed_frames, 0)
    job.progress_updated_at = datetime.utcnow()
    if commit:
        db.commit()
        db.refresh(job)
    _publish_job_event(job, event_type=event_type)


def _mark_failed(db: Session, job: ReconstructionJob, message: str) -> None:
    job.status = "failed"
    job.error_message = message
    job.progress_message = message
    job.progress_stage = "failed"
    job.progress_percent = 100
    job.finished_at = datetime.utcnow()
    _update_job_progress(db, job, event_type="failed", commit=True)


def _mark_cancelled(db: Session, job: ReconstructionJob, message: str = "cancelled by user") -> None:
    job.status = "cancelled"
    job.error_message = None
    job.progress_message = message
    job.progress_stage = "cancelled"
    job.progress_percent = 100
    job.finished_at = datetime.utcnow()
    _update_job_progress(db, job, event_type="cancelled", commit=True)


def _process_job(job_id: str) -> None:
    db: Session = SessionLocal()
    try:
        job: Optional[ReconstructionJob] = db.query(ReconstructionJob).filter(ReconstructionJob.id == job_id).first()
        if job is None:
            return

        if job.status == "cancelled":
            if job.finished_at is None:
                _mark_cancelled(db, job, message=job.progress_message or "cancelled before start")
            else:
                _publish_job_event(job, event_type="cancelled")
            return

        _update_job_progress(
            db,
            job,
            percent=_clamp_percent(job.progress_percent, 0),
            stage=job.progress_stage or "queued",
            message=job.progress_message or "job queued",
            event_type="queued",
            commit=True,
        )

        media: Optional[MediaAsset] = db.query(MediaAsset).filter(MediaAsset.id == job.media_id).first()
        if media is None:
            _mark_failed(db, job, "Media not found")
            return

        job.status = "running"
        job.started_at = datetime.utcnow()
        _update_job_progress(
            db,
            job,
            percent=max(1, _clamp_percent(job.progress_percent, 0)),
            stage="running",
            message="job started",
            event_type="progress",
            commit=True,
        )

        output_dir = create_job_output_dir(job.user_id, job.id)

        progress_state = {"last_percent": _clamp_percent(job.progress_percent, 0), "last_ts": 0.0}
        cancel_state = {"last_db_check": 0.0}

        def on_progress(event: Dict[str, Any]) -> None:
            if _has_cancel_request(job_id):
                return
            now = time.time()
            percent = event.get("percent")
            if percent is None:
                percent = progress_state["last_percent"]
            clamped = _clamp_percent(percent, progress_state["last_percent"])
            delta = clamped - progress_state["last_percent"]
            if clamped < 100 and delta <= 0 and (now - progress_state["last_ts"] < 0.8):
                return
            progress_state["last_percent"] = clamped
            progress_state["last_ts"] = now
            _update_job_progress(
                db,
                job,
                percent=clamped,
                stage=event.get("stage"),
                message=event.get("message"),
                total_frames=event.get("total_frames"),
                processed_frames=event.get("processed_frames"),
                event_type="progress",
                commit=True,
            )

        def should_abort() -> bool:
            if _has_cancel_request(job_id):
                return True
            now = time.time()
            if now - cancel_state["last_db_check"] < 0.9:
                return False
            cancel_state["last_db_check"] = now
            status = db.query(ReconstructionJob.status).filter(ReconstructionJob.id == job_id).scalar()
            if status == "cancelled":
                request_job_cancel(job_id)
                return True
            return False

        result = run_reconstruction(
            media_type=media.media_type,
            media_path=media.stored_path,
            output_dir=output_dir,
            stem="reconstruction",
            progress_callback=on_progress,
            should_abort=should_abort,
            attention_scenario=str(job.attention_scenario or "classroom"),
        )

        if should_abort():
            raise JobCancelledError("Job cancelled by user")

        job.status = "completed"
        job.error_message = None
        job.output_model_path = result.get("model_path")
        job.output_preview_path = result.get("preview_path")
        job.output_sequence_zip_path = result.get("sequence_zip_path")
        job.output_animation_path = result.get("animation_path")
        job.output_metadata_path = result.get("metadata_path")
        job.output_attention_metadata_path = result.get("attention_metadata_path")
        job.keyframe_index = result.get("keyframe_index")
        job.log_text = result.get("log_text")
        job.finished_at = datetime.utcnow()
        _update_job_progress(
            db,
            job,
            percent=100,
            stage="completed",
            message="job completed",
            event_type="completed",
            commit=True,
        )
    except JobCancelledError:
        cancelled_job: Optional[ReconstructionJob] = db.query(ReconstructionJob).filter(ReconstructionJob.id == job_id).first()
        if cancelled_job is not None:
            _mark_cancelled(db, cancelled_job)
    except Exception as exc:  # noqa: BLE001
        failed_job: Optional[ReconstructionJob] = db.query(ReconstructionJob).filter(ReconstructionJob.id == job_id).first()
        if failed_job is not None:
            _mark_failed(db, failed_job, str(exc))
    finally:
        _clear_cancel_request(job_id)
        db.close()


def submit_job(job_id: str) -> None:
    future = _executor.submit(_process_job, job_id)
    _register_job_future(job_id, future)


def recover_orphaned_jobs() -> dict[str, int]:
    db: Session = SessionLocal()
    summary = {"requeued": 0, "cancelled": 0, "failed": 0}
    now = datetime.utcnow()
    try:
        queued_jobs = (
            db.query(ReconstructionJob)
            .filter(ReconstructionJob.status == "queued")
            .all()
        )
        for job in queued_jobs:
            if is_job_active(job.id):
                continue
            submit_job(job.id)
            summary["requeued"] += 1

        running_jobs = (
            db.query(ReconstructionJob)
            .filter(ReconstructionJob.status == "running")
            .all()
        )
        for job in running_jobs:
            if is_job_active(job.id):
                continue
            if str(job.progress_stage or "") == "cancel_requested":
                job.status = "cancelled"
                job.progress_stage = "cancelled"
                job.progress_message = "cancelled after backend restart"
                job.progress_percent = 100
                job.error_message = None
                job.finished_at = now
                job.progress_updated_at = now
                summary["cancelled"] += 1
            else:
                message = "job interrupted because backend restarted"
                job.status = "failed"
                job.progress_stage = "failed"
                job.progress_message = message
                job.progress_percent = 100
                job.error_message = message
                job.finished_at = now
                job.progress_updated_at = now
                summary["failed"] += 1

        db.commit()
        return summary
    finally:
        db.close()


def shutdown_job_executor() -> None:
    _executor.shutdown(wait=False)
