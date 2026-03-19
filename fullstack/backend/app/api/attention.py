from __future__ import annotations

from datetime import datetime, timedelta
import threading
from typing import Dict, Optional

from fastapi import APIRouter, File, HTTPException, Query, UploadFile, status

from app.core.schemas import RealtimeAttentionFace, RealtimeAttentionResponse
from app.services.reconstruction import analyze_attention_frame


router = APIRouter(prefix="/attention", tags=["attention"])

_state_lock = threading.Lock()
_live_state: Dict[str, Dict[str, float]] = {}


def _cleanup_states() -> None:
    now = datetime.utcnow()
    expire_before = now - timedelta(minutes=10)
    with _state_lock:
        for key in list(_live_state.keys()):
            ts = _live_state[key].get("updated_ts")
            if ts is None or datetime.fromtimestamp(ts) < expire_before:
                _live_state.pop(key, None)


def _apply_single_face_smoothing(session_id: str, face: Dict[str, object], alpha: float) -> Dict[str, object]:
    a = max(0.0, min(0.95, float(alpha)))
    now_ts = datetime.utcnow().timestamp()
    with _state_lock:
        prev = _live_state.get(session_id)
        if prev is None:
            _live_state[session_id] = {
                "yaw": float(face.get("yaw", 0.0)),
                "pitch": float(face.get("pitch", 0.0)),
                "roll": float(face.get("roll", 0.0)),
                "attention_score": float(face.get("attention_score", 0.0)),
                "updated_ts": now_ts,
            }
            return face

        smoothed = {
            "yaw": a * float(prev["yaw"]) + (1.0 - a) * float(face.get("yaw", 0.0)),
            "pitch": a * float(prev["pitch"]) + (1.0 - a) * float(face.get("pitch", 0.0)),
            "roll": a * float(prev["roll"]) + (1.0 - a) * float(face.get("roll", 0.0)),
            "attention_score": a * float(prev["attention_score"]) + (1.0 - a) * float(face.get("attention_score", 0.0)),
        }
        prev.update(smoothed)
        prev["updated_ts"] = now_ts

    out = dict(face)
    out["yaw"] = round(float(smoothed["yaw"]), 4)
    out["pitch"] = round(float(smoothed["pitch"]), 4)
    out["roll"] = round(float(smoothed["roll"]), 4)
    out["attention_score"] = round(float(smoothed["attention_score"]), 4)
    return out


@router.post("/frame", response_model=RealtimeAttentionResponse)
def analyze_frame(
    file: UploadFile = File(...),
    scenario: str = Query(default="classroom", pattern="^(classroom|exam|driving)$"),
    mode: str = Query(default="single", pattern="^(single|multi)$"),
    session_id: Optional[str] = Query(default=None, min_length=3, max_length=64),
    smoothing_alpha: float = Query(default=0.72, ge=0.0, le=0.95),
) -> RealtimeAttentionResponse:
    image_bytes = file.file.read()
    if not image_bytes:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Empty image file")

    try:
        analyzed = analyze_attention_frame(
            image_bytes,
            scenario=scenario,
            mode=mode,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    faces_raw = list(analyzed.get("faces", []))
    if mode == "single" and session_id and faces_raw:
        faces_raw[0] = _apply_single_face_smoothing(session_id, faces_raw[0], smoothing_alpha)

    face_models = [
        RealtimeAttentionFace(
            face_index=int(face.get("face_index", idx)),
            yaw=float(face.get("yaw", 0.0)),
            pitch=float(face.get("pitch", 0.0)),
            roll=float(face.get("roll", 0.0)),
            attention_score=float(face.get("attention_score", 0.0)),
            head_down=bool(face.get("head_down", False)),
            side_view=bool(face.get("side_view", False)),
            tilted=bool(face.get("tilted", False)),
            distracted=bool(face.get("distracted", False)),
        )
        for idx, face in enumerate(faces_raw)
    ]

    face_count = len(face_models)
    avg_attention = 0.0
    head_up_rate = 0.0
    if face_count > 0:
        avg_attention = sum(face.attention_score for face in face_models) / float(face_count)
        head_up_rate = (
            sum(1 for face in face_models if (not face.head_down and not face.side_view)) / float(face_count)
        ) * 100.0

    _cleanup_states()
    return RealtimeAttentionResponse(
        mode="multi" if mode == "multi" else "single",
        scenario=scenario,
        timestamp=datetime.utcnow(),
        face_count=face_count,
        avg_attention=round(avg_attention, 4),
        classroom_head_up_rate=round(head_up_rate, 4),
        faces=face_models,
    )
