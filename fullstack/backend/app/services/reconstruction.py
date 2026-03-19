from __future__ import annotations

import json
import os
import sys
import threading
import zipfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml

from app.core.config import settings


ProgressCallback = Optional[Callable[[Dict[str, Any]], None]]
CancelCheck = Optional[Callable[[], bool]]


SCENARIO_CONFIG: Dict[str, Dict[str, float]] = {
    "classroom": {
        "yaw_warn": 18.0,
        "yaw_severe": 32.0,
        "pitch_down_warn": 14.0,
        "pitch_down_severe": 24.0,
        "pitch_up_warn": 18.0,
        "roll_warn": 18.0,
        "roll_severe": 30.0,
        "low_attention_threshold": 60.0,
        "rapid_turn_delta": 26.0,
    },
    "exam": {
        "yaw_warn": 15.0,
        "yaw_severe": 26.0,
        "pitch_down_warn": 12.0,
        "pitch_down_severe": 20.0,
        "pitch_up_warn": 15.0,
        "roll_warn": 14.0,
        "roll_severe": 24.0,
        "low_attention_threshold": 66.0,
        "rapid_turn_delta": 22.0,
    },
    "driving": {
        "yaw_warn": 14.0,
        "yaw_severe": 24.0,
        "pitch_down_warn": 10.0,
        "pitch_down_severe": 18.0,
        "pitch_up_warn": 16.0,
        "roll_warn": 12.0,
        "roll_severe": 22.0,
        "low_attention_threshold": 68.0,
        "rapid_turn_delta": 20.0,
    },
}


class JobCancelledError(RuntimeError):
    pass


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _scenario_key(value: Optional[str]) -> str:
    key = (value or "classroom").strip().lower()
    if key not in SCENARIO_CONFIG:
        return "classroom"
    return key


def _report_progress(progress_callback: ProgressCallback, **event: Any) -> None:
    if progress_callback is None:
        return
    try:
        progress_callback(event)
    except Exception:
        return


def _check_cancel(should_abort: CancelCheck) -> None:
    if should_abort is None:
        return
    try:
        if should_abort():
            raise JobCancelledError("Job cancelled by user")
    except JobCancelledError:
        raise
    except Exception:
        return


def _ema(prev: float, cur: float, alpha: float) -> float:
    return alpha * prev + (1.0 - alpha) * cur


def _interpolate_pose(prev_pose: Dict[str, float], blend: float = 0.1) -> Dict[str, float]:
    b = _clamp(blend, 0.0, 0.3)
    return {
        "yaw": prev_pose.get("yaw", 0.0) * (1.0 - b),
        "pitch": prev_pose.get("pitch", 0.0) * (1.0 - b),
        "roll": prev_pose.get("roll", 0.0) * (1.0 - b),
    }


def _attention_from_pose(yaw: float, pitch: float, roll: float, scenario: str) -> Tuple[float, Dict[str, bool]]:
    cfg = SCENARIO_CONFIG[_scenario_key(scenario)]
    abs_yaw = abs(yaw)
    abs_pitch = abs(pitch)
    abs_roll = abs(roll)

    side_view = abs_yaw >= cfg["yaw_warn"]
    head_down = pitch >= cfg["pitch_down_warn"]
    tilted = abs_roll >= cfg["roll_warn"]

    penalty = 0.0
    if abs_yaw > cfg["yaw_warn"]:
        penalty += min(35.0, (abs_yaw - cfg["yaw_warn"]) * 1.4)
    if abs_yaw > cfg["yaw_severe"]:
        penalty += min(25.0, (abs_yaw - cfg["yaw_severe"]) * 1.0)

    if pitch > cfg["pitch_down_warn"]:
        penalty += min(35.0, (pitch - cfg["pitch_down_warn"]) * 1.8)
    if pitch > cfg["pitch_down_severe"]:
        penalty += min(20.0, (pitch - cfg["pitch_down_severe"]) * 1.3)
    if -pitch > cfg["pitch_up_warn"]:
        penalty += min(18.0, (-pitch - cfg["pitch_up_warn"]) * 1.0)

    if abs_roll > cfg["roll_warn"]:
        penalty += min(24.0, (abs_roll - cfg["roll_warn"]) * 1.2)
    if abs_roll > cfg["roll_severe"]:
        penalty += min(14.0, (abs_roll - cfg["roll_severe"]) * 1.0)

    if head_down and side_view:
        penalty += 8.0

    score = _clamp(100.0 - penalty, 0.0, 100.0)
    distracted = score < cfg["low_attention_threshold"]

    return score, {
        "head_down": head_down,
        "side_view": side_view,
        "tilted": tilted,
        "distracted": distracted,
    }


def _longest_true_run(entries: List[Dict[str, object]], key: str) -> int:
    longest = 0
    cur = 0
    for entry in entries:
        if bool(entry.get(key)):
            cur += 1
            if cur > longest:
                longest = cur
        else:
            cur = 0
    return longest


def _build_attention_summary(
    entries: List[Dict[str, object]],
    fps: float,
    detected_frames: int,
    interpolated_frames: int,
    scenario: str,
    rapid_turn_events: int,
) -> Dict[str, object]:
    if not entries:
        return {
            "scenario": _scenario_key(scenario),
            "fps": round(float(fps), 4),
            "total_frames": 0,
            "detected_frames": int(detected_frames),
            "interpolated_frames": int(interpolated_frames),
            "avg_attention": 0.0,
            "min_attention": 0.0,
            "max_attention": 0.0,
            "low_attention_ratio": 0.0,
            "head_down_ratio": 0.0,
            "side_view_ratio": 0.0,
            "rapid_turn_events": int(rapid_turn_events),
            "longest_distracted_frames": 0,
            "classroom_head_up_rate": 0.0,
            "exam_focus_score": 0.0,
            "driving_risk_score": 0.0,
            "warnings": ["no-face-detected"],
        }

    scores = [float(e.get("attention_score", 0.0)) for e in entries]
    total = len(entries)
    low_ratio = sum(1 for s in scores if s < 60.0) / float(total)
    head_down_ratio = sum(1 for e in entries if e.get("head_down")) / float(total)
    side_view_ratio = sum(1 for e in entries if e.get("side_view")) / float(total)
    rapid_turn_ratio = rapid_turn_events / float(max(1, total))

    exam_focus_score = _clamp(100.0 - (low_ratio * 55.0 + head_down_ratio * 20.0 + side_view_ratio * 25.0) * 100.0, 0.0, 100.0)
    driving_risk_score = _clamp((low_ratio * 65.0 + rapid_turn_ratio * 20.0 + side_view_ratio * 15.0) * 100.0, 0.0, 100.0)

    warnings: List[str] = []
    if low_ratio > 0.35:
        warnings.append("frequent-low-attention")
    if head_down_ratio > 0.28:
        warnings.append("head-down-too-long")
    if side_view_ratio > 0.3:
        warnings.append("long-side-view")
    if rapid_turn_events >= max(3, int(round((total / max(1.0, fps)) / 10.0))):
        warnings.append("frequent-head-turn")
    if driving_risk_score >= 55.0:
        warnings.append("driving-distraction-risk")

    return {
        "scenario": _scenario_key(scenario),
        "fps": round(float(fps), 4),
        "total_frames": total,
        "detected_frames": int(detected_frames),
        "interpolated_frames": int(interpolated_frames),
        "avg_attention": round(float(np.mean(scores)), 4),
        "min_attention": round(float(min(scores)), 4),
        "max_attention": round(float(max(scores)), 4),
        "low_attention_ratio": round(float(low_ratio), 4),
        "head_down_ratio": round(float(head_down_ratio), 4),
        "side_view_ratio": round(float(side_view_ratio), 4),
        "rapid_turn_events": int(rapid_turn_events),
        "longest_distracted_frames": int(_longest_true_run(entries, "distracted")),
        "classroom_head_up_rate": round((1.0 - head_down_ratio) * 100.0, 3),
        "exam_focus_score": round(float(exam_focus_score), 3),
        "driving_risk_score": round(float(driving_risk_score), 3),
        "warnings": warnings,
    }


class ThreeDDFARunner:
    """Lazy-initialized 3DDFA inference runner."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._is_ready = False
        self.face_boxes = None
        self.tddfa = None
        self.render = None
        self.ser_to_obj = None

    def _ensure_project_on_path(self) -> None:
        project_str = str(settings.project_root)
        if project_str not in sys.path:
            sys.path.insert(0, project_str)

    def _init_model(self) -> None:
        with self._lock:
            if self._is_ready:
                return

            self._ensure_project_on_path()
            cfg_path = settings.project_root / "configs" / "mb1_120x120.yml"
            cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
            for key in ("checkpoint_fp", "bfm_fp", "onnx_fp", "param_mean_std_fp"):
                value = cfg.get(key)
                if isinstance(value, str) and not os.path.isabs(value):
                    cfg[key] = str(settings.project_root / value)

            if settings.use_onnx:
                os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
                os.environ["OMP_NUM_THREADS"] = str(settings.inference_threads_per_worker)
                os.environ["MKL_NUM_THREADS"] = str(settings.inference_threads_per_worker)
                from FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX
                from TDDFA_ONNX import TDDFA_ONNX

                self.face_boxes = FaceBoxes_ONNX()
                self.tddfa = TDDFA_ONNX(**cfg)
            else:
                from FaceBoxes import FaceBoxes
                from TDDFA import TDDFA

                self.face_boxes = FaceBoxes()
                self.tddfa = TDDFA(gpu_mode=False, **cfg)

            from utils.render import render
            from utils.serialization import ser_to_obj

            self.render = render
            self.ser_to_obj = ser_to_obj
            self._is_ready = True

    @staticmethod
    def _box_area(box) -> float:
        return float(max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1]))

    @staticmethod
    def _box_score(box) -> float:
        try:
            if len(box) > 4:
                return float(box[4])
        except Exception:
            return 0.0
        return 0.0

    @staticmethod
    def _box_iou(box_a, box_b) -> float:
        ax1, ay1, ax2, ay2 = [float(v) for v in box_a[:4]]
        bx1, by1, bx2, by2 = [float(v) for v in box_b[:4]]

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter = inter_w * inter_h
        if inter <= 0.0:
            return 0.0

        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        denom = area_a + area_b - inter
        if denom <= 0.0:
            return 0.0
        return inter / denom

    def _filter_face_boxes(
        self,
        boxes,
        frame_shape,
        *,
        min_score: float,
        min_area_ratio: float,
        min_side: float,
        max_faces: int,
        nms_iou: float = 0.35,
    ):
        height, width = frame_shape[:2]
        max_x = max(0.0, float(width - 1))
        max_y = max(0.0, float(height - 1))
        frame_area = max(1.0, float(width * height))

        def _collect(score_floor: float, area_floor: float, side_floor: float):
            selected = []
            for box in boxes:
                if box is None or len(box) < 4:
                    continue
                x1 = _clamp(_safe_float(box[0]), 0.0, max_x)
                y1 = _clamp(_safe_float(box[1]), 0.0, max_y)
                x2 = _clamp(_safe_float(box[2]), 0.0, max_x)
                y2 = _clamp(_safe_float(box[3]), 0.0, max_y)
                bw = x2 - x1
                bh = y2 - y1
                if bw <= 1.0 or bh <= 1.0:
                    continue
                if bw < side_floor or bh < side_floor:
                    continue

                score = self._box_score(box)
                if score < score_floor:
                    continue

                area = bw * bh
                if area / frame_area < area_floor:
                    continue

                selected.append([x1, y1, x2, y2, score])
            return selected

        filtered = _collect(min_score, min_area_ratio, min_side)
        if not filtered:
            filtered = _collect(
                max(0.12, min_score * 0.72),
                max(0.00005, min_area_ratio * 0.35),
                max(8.0, min_side * 0.75),
            )

        filtered.sort(key=lambda b: (float(b[4]), self._box_area(b)), reverse=True)
        deduped = []
        max_keep = max(1, int(max_faces))
        for box in filtered:
            if any(self._box_iou(box, kept) >= nms_iou for kept in deduped):
                continue
            deduped.append(box)
            if len(deduped) >= max_keep:
                break

        deduped.sort(key=self._box_area, reverse=True)
        return deduped

    def _largest_face_box(self, frame):
        boxes = self._detect_face_boxes(frame, profile="photo")
        if len(boxes) == 0:
            return None, 0.0
        largest_box = max(boxes, key=self._box_area)
        return largest_box, self._box_area(largest_box)

    def _detect_face_boxes(self, frame, *, profile: str = "video"):
        profile_key = str(profile).strip().lower()
        boxes = list(self.face_boxes(frame))
        candidates = list(boxes)

        # For still images/realtime frames, upscale once to recover small faces.
        if profile_key in {"photo", "realtime"} and len(candidates) <= 4:
            height, width = frame.shape[:2]
            scales = (1.35, 1.7) if max(height, width) <= 1920 else (1.2, 1.45)
            for scale in scales:
                resized = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
                scaled_boxes = list(self.face_boxes(resized))
                if not scaled_boxes:
                    continue
                inv = 1.0 / float(scale)
                for box in scaled_boxes:
                    if box is None or len(box) < 4:
                        continue
                    candidates.append(
                        [
                            _safe_float(box[0]) * inv,
                            _safe_float(box[1]) * inv,
                            _safe_float(box[2]) * inv,
                            _safe_float(box[3]) * inv,
                            self._box_score(box),
                        ]
                    )

        if profile_key == "photo":
            return self._filter_face_boxes(
                candidates,
                frame.shape,
                min_score=0.34,
                min_area_ratio=0.00016,
                min_side=12.0,
                max_faces=18,
                nms_iou=0.33,
            )
        if profile_key == "realtime":
            return self._filter_face_boxes(
                candidates,
                frame.shape,
                min_score=0.38,
                min_area_ratio=0.0002,
                min_side=12.0,
                max_faces=14,
                nms_iou=0.33,
            )
        return self._filter_face_boxes(
            candidates,
            frame.shape,
            min_score=0.55,
            min_area_ratio=0.0008,
            min_side=18.0,
            max_faces=6,
            nms_iou=0.35,
        )

    def _infer_face(self, frame, face_box) -> Tuple[object, object]:
        param_lst, roi_box_lst = self.tddfa(frame, [face_box])
        ver_lst = self.tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=True)
        return param_lst[0], ver_lst[0]

    def _infer_faces(self, frame, face_boxes) -> Tuple[List[object], List[object]]:
        param_lst, roi_box_lst = self.tddfa(frame, face_boxes)
        ver_lst = self.tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=True)
        return list(param_lst), list(ver_lst)

    def _pose_from_param(self, param) -> Tuple[float, float, float]:
        self._ensure_project_on_path()
        from utils.pose import calc_pose

        _, pose = calc_pose(param)
        yaw, pitch, roll = [float(v) for v in pose]
        return yaw, pitch, roll

    def _save_obj_and_preview(
        self,
        frame,
        ver_lst,
        model_path: Path,
        preview_path: Path,
    ) -> None:
        self.ser_to_obj(frame, ver_lst, self.tddfa.tri, height=frame.shape[0], wfp=str(model_path))
        self.render(frame.copy(), ver_lst, self.tddfa.tri, alpha=0.6, show_flag=False, wfp=str(preview_path))

    def _analyze_faces(self, frame, scenario: str, mode: str) -> Dict[str, object]:
        self._init_model()
        mode_key = "multi" if str(mode).strip().lower() == "multi" else "single"
        boxes = self._detect_face_boxes(frame, profile="realtime")
        if len(boxes) == 0:
            return {
                "mode": mode_key,
                "scenario": _scenario_key(scenario),
                "face_count": 0,
                "avg_attention": 0.0,
                "classroom_head_up_rate": 0.0,
                "faces": [],
            }

        if mode_key == "single":
            largest = max(boxes, key=self._box_area)
            boxes = [largest]

        param_lst, _ = self.tddfa(frame, boxes)
        faces = []
        head_up = 0
        score_sum = 0.0

        for idx, param in enumerate(param_lst):
            yaw, pitch, roll = self._pose_from_param(param)
            score, flags = _attention_from_pose(yaw, pitch, roll, scenario)
            faces.append(
                {
                    "face_index": idx,
                    "yaw": round(yaw, 4),
                    "pitch": round(pitch, 4),
                    "roll": round(roll, 4),
                    "attention_score": round(score, 4),
                    "head_down": bool(flags["head_down"]),
                    "side_view": bool(flags["side_view"]),
                    "tilted": bool(flags["tilted"]),
                    "distracted": bool(flags["distracted"]),
                }
            )
            score_sum += score
            if (not flags["head_down"]) and (not flags["side_view"]):
                head_up += 1

        count = len(faces)
        return {
            "mode": mode_key,
            "scenario": _scenario_key(scenario),
            "face_count": count,
            "avg_attention": round(score_sum / max(1, count), 4),
            "classroom_head_up_rate": round((head_up / max(1, count)) * 100.0, 4),
            "faces": faces,
        }

    def reconstruct_photo(
        self,
        photo_path: str,
        output_dir: Path,
        stem: str,
        progress_callback: ProgressCallback = None,
        should_abort: CancelCheck = None,
        attention_scenario: str = "classroom",
    ) -> Dict[str, Optional[str]]:
        self._init_model()
        _check_cancel(should_abort)
        _report_progress(progress_callback, stage="photo_load", percent=8, message="loading image")

        frame = cv2.imread(photo_path)
        if frame is None:
            raise RuntimeError("Failed to read image file.")

        _check_cancel(should_abort)
        _report_progress(progress_callback, stage="photo_detect", percent=28, message="detecting face")
        face_boxes = self._detect_face_boxes(frame, profile="photo")
        if not face_boxes:
            raise RuntimeError("No face detected in image.")

        param_lst, ver_lst = self._infer_faces(frame, face_boxes)
        photo_faces: List[Dict[str, object]] = []
        attention_entries: List[Dict[str, object]] = []

        for idx, (face_box, param) in enumerate(zip(face_boxes, param_lst)):
            yaw, pitch, roll = self._pose_from_param(param)
            attention_score, flags = _attention_from_pose(yaw, pitch, roll, attention_scenario)
            detection_score = float(face_box[4]) if len(face_box) > 4 else 0.0
            photo_faces.append(
                {
                    "face_index": idx,
                    "yaw": round(yaw, 4),
                    "pitch": round(pitch, 4),
                    "roll": round(roll, 4),
                    "attention_score": round(attention_score, 4),
                    "head_down": bool(flags["head_down"]),
                    "side_view": bool(flags["side_view"]),
                    "tilted": bool(flags["tilted"]),
                    "distracted": bool(flags["distracted"]),
                    "detection_score": round(detection_score, 4),
                }
            )
            attention_entries.append(
                {
                    "frame_index": idx,
                    "yaw": round(yaw, 4),
                    "pitch": round(pitch, 4),
                    "roll": round(roll, 4),
                    "attention_score": round(attention_score, 4),
                    "source": "detected",
                    "detection_score": round(detection_score, 4),
                    "head_down": bool(flags["head_down"]),
                    "side_view": bool(flags["side_view"]),
                    "tilted": bool(flags["tilted"]),
                    "distracted": bool(flags["distracted"]),
                    "rapid_turn": False,
                }
            )

        _check_cancel(should_abort)
        _report_progress(progress_callback, stage="photo_mesh", percent=62, message="generating mesh")

        model_path = output_dir / "{}.obj".format(stem)
        preview_path = output_dir / "{}.jpg".format(stem)
        self._save_obj_and_preview(frame, ver_lst, model_path, preview_path)

        attention_summary = _build_attention_summary(
            attention_entries,
            fps=1.0,
            detected_frames=len(attention_entries),
            interpolated_frames=0,
            scenario=attention_scenario,
            rapid_turn_events=0,
        )
        attention_metadata_path = output_dir / "{}_attention.json".format(stem)
        attention_metadata_path.write_text(
            json.dumps(
                {
                    "media_type": "photo",
                    "mode": "multi" if len(photo_faces) > 1 else "single",
                    "scenario": _scenario_key(attention_scenario),
                    "fps": 1.0,
                    "summary": attention_summary,
                    "faces": photo_faces,
                    "entries": attention_entries,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        _check_cancel(should_abort)
        _report_progress(progress_callback, stage="photo_done", percent=100, message="photo reconstruction completed")

        return {
            "model_path": str(model_path),
            "preview_path": str(preview_path),
            "sequence_zip_path": None,
            "animation_path": None,
            "metadata_path": None,
            "attention_metadata_path": str(attention_metadata_path),
            "keyframe_index": None,
            "log_text": "Image reconstructed with multi-face 3D reconstruction and head-pose attention analysis. detected_faces={}".format(len(photo_faces)),
        }

    def reconstruct_video(
        self,
        video_path: str,
        output_dir: Path,
        stem: str,
        progress_callback: ProgressCallback = None,
        should_abort: CancelCheck = None,
        attention_scenario: str = "classroom",
    ) -> Dict[str, Optional[str]]:
        self._init_model()
        _check_cancel(should_abort)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError("Failed to open video file.")

        sequence_dir = output_dir / "sequence_obj"
        sequence_dir.mkdir(parents=True, exist_ok=True)
        preview_dir = output_dir / "sequence_preview"
        preview_dir.mkdir(parents=True, exist_ok=True)

        animation_path = output_dir / "animation.mp4"
        metadata_path = output_dir / "sequence_metadata.json"
        sequence_zip_path = output_dir / "sequence_obj.zip"
        attention_metadata_path = output_dir / "attention_metadata.json"

        writer = None
        frame_index = 0
        total_frames = 0
        reconstructed_frames = 0
        interpolated_frames = 0
        rapid_turn_events = 0
        sequence_entries: List[Dict[str, object]] = []
        attention_entries: List[Dict[str, object]] = []

        best_score = -1.0
        best_face_count = 0
        best_total_area = 0.0
        best_frame = None
        best_ver_lst = None
        best_index = -1

        scenario_key = _scenario_key(attention_scenario)
        rapid_turn_delta = SCENARIO_CONFIG[scenario_key]["rapid_turn_delta"]
        smoothed_pose = {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}
        smoothed_score = 100.0
        has_pose = False

        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps is None or fps <= 0:
                fps = 25.0
            total_frame_hint = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            _report_progress(
                progress_callback,
                stage="video_prepare",
                percent=4,
                message="video opened",
                total_frames=total_frame_hint if total_frame_hint > 0 else None,
                processed_frames=0,
            )

            while True:
                _check_cancel(should_abort)
                ok, frame = cap.read()
                if not ok:
                    break

                total_frames += 1
                if writer is None:
                    h, w = frame.shape[:2]
                    writer = cv2.VideoWriter(
                        str(animation_path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        fps,
                        (w, h),
                    )

                face_boxes = self._detect_face_boxes(frame, profile="video")
                source = "detected"
                detection_score = 0.0
                rapid_turn = False

                if not face_boxes:
                    writer.write(frame)
                    raw_pose = _interpolate_pose(smoothed_pose if has_pose else {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}, blend=0.12)
                    source = "interpolated"
                    detection_score = 0.0
                    interpolated_frames += 1
                else:
                    param_lst, ver_lst = self._infer_faces(frame, face_boxes)
                    dominant_index = max(range(len(face_boxes)), key=lambda idx: self._box_area(face_boxes[idx]))
                    dominant_param = param_lst[dominant_index]
                    raw_yaw, raw_pitch, raw_roll = self._pose_from_param(dominant_param)
                    raw_pose = {"yaw": raw_yaw, "pitch": raw_pitch, "roll": raw_roll}
                    reconstructed_frames += 1
                    detection_score = sum(float(box[4]) if len(box) > 4 else 0.0 for box in face_boxes) / float(max(1, len(face_boxes)))
                    total_area = sum(self._box_area(box) for box in face_boxes)

                    obj_name = "frame_{:06d}.obj".format(frame_index)
                    obj_path = sequence_dir / obj_name
                    self.ser_to_obj(frame, ver_lst, self.tddfa.tri, height=frame.shape[0], wfp=str(obj_path))

                    rendered = self.render(frame.copy(), ver_lst, self.tddfa.tri, alpha=0.6, show_flag=False)
                    preview_name = "frame_{:06d}.jpg".format(frame_index)
                    preview_path = preview_dir / preview_name
                    cv2.imwrite(str(preview_path), rendered)
                    writer.write(rendered)

                    sequence_entries.append(
                        {
                            "frame_index": frame_index,
                            "face_count": len(face_boxes),
                            "obj_file": obj_name,
                            "preview_file": preview_name,
                        }
                    )

                    if (
                        len(face_boxes) > best_face_count
                        or (len(face_boxes) == best_face_count and total_area > best_total_area)
                        or (len(face_boxes) == best_face_count and abs(total_area - best_total_area) < 1e-6 and detection_score > best_score)
                    ):
                        best_score = detection_score
                        best_face_count = len(face_boxes)
                        best_total_area = total_area
                        best_frame = frame.copy()
                        best_ver_lst = [ver.copy() for ver in ver_lst]
                        best_index = frame_index

                raw_score, raw_flags = _attention_from_pose(
                    raw_pose["yaw"],
                    raw_pose["pitch"],
                    raw_pose["roll"],
                    scenario_key,
                )

                if not has_pose:
                    smoothed_pose = dict(raw_pose)
                    smoothed_score = raw_score
                    has_pose = True
                else:
                    prev_yaw = smoothed_pose["yaw"]
                    smoothed_pose["yaw"] = _ema(smoothed_pose["yaw"], raw_pose["yaw"], alpha=0.72)
                    smoothed_pose["pitch"] = _ema(smoothed_pose["pitch"], raw_pose["pitch"], alpha=0.72)
                    smoothed_pose["roll"] = _ema(smoothed_pose["roll"], raw_pose["roll"], alpha=0.72)
                    smoothed_score = _ema(smoothed_score, raw_score, alpha=0.78)
                    rapid_turn = source == "detected" and abs(smoothed_pose["yaw"] - prev_yaw) >= rapid_turn_delta
                    if rapid_turn:
                        rapid_turn_events += 1

                smooth_score, smooth_flags = _attention_from_pose(
                    smoothed_pose["yaw"],
                    smoothed_pose["pitch"],
                    smoothed_pose["roll"],
                    scenario_key,
                )
                blended_score = _clamp(0.35 * smooth_score + 0.65 * smoothed_score, 0.0, 100.0)
                smooth_flags["distracted"] = blended_score < SCENARIO_CONFIG[scenario_key]["low_attention_threshold"]

                attention_entries.append(
                    {
                        "frame_index": frame_index,
                        "yaw": round(float(smoothed_pose["yaw"]), 4),
                        "pitch": round(float(smoothed_pose["pitch"]), 4),
                        "roll": round(float(smoothed_pose["roll"]), 4),
                        "attention_score": round(float(blended_score), 4),
                        "source": source,
                        "detection_score": round(float(detection_score), 4),
                        "head_down": bool(smooth_flags["head_down"]),
                        "side_view": bool(smooth_flags["side_view"]),
                        "tilted": bool(smooth_flags["tilted"]),
                        "distracted": bool(smooth_flags["distracted"]),
                        "rapid_turn": bool(rapid_turn),
                    }
                )

                frame_index += 1
                remaining_hint = total_frame_hint - frame_index if total_frame_hint > 0 else None
                if frame_index % 3 == 0 or (remaining_hint is not None and remaining_hint <= 2):
                    if total_frame_hint > 0:
                        ratio = frame_index / float(max(1, total_frame_hint))
                        progress = min(95, max(5, int(round(ratio * 90.0)) + 5))
                    else:
                        progress = min(95, 5 + int(frame_index * 0.25))
                    _report_progress(
                        progress_callback,
                        stage="video_processing",
                        percent=progress,
                        message="processing video frames",
                        total_frames=total_frame_hint if total_frame_hint > 0 else None,
                        processed_frames=frame_index,
                        detected_frames=reconstructed_frames,
                        interpolated_frames=interpolated_frames,
                    )
        finally:
            cap.release()
            if writer is not None:
                writer.release()

        if reconstructed_frames == 0 or best_frame is None or best_ver_lst is None:
            raise RuntimeError("No face detected in video frames.")

        _check_cancel(should_abort)
        _report_progress(
            progress_callback,
            stage="video_finalize",
            percent=96,
            message="finalizing keyframe",
            total_frames=total_frames,
            processed_frames=total_frames,
            detected_frames=reconstructed_frames,
            interpolated_frames=interpolated_frames,
        )
        model_path = output_dir / "{}_keyframe.obj".format(stem)
        preview_path = output_dir / "{}_keyframe.jpg".format(stem)
        self._save_obj_and_preview(best_frame, best_ver_lst, model_path, preview_path)

        _check_cancel(should_abort)
        _report_progress(
            progress_callback,
            stage="video_packaging",
            percent=97,
            message="packaging frame models",
            total_frames=total_frames,
            processed_frames=total_frames,
            detected_frames=reconstructed_frames,
            interpolated_frames=interpolated_frames,
        )
        total_sequence_entries = len(sequence_entries)
        report_every = max(1, total_sequence_entries // 24) if total_sequence_entries > 0 else 1
        with zipfile.ZipFile(
            sequence_zip_path,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=1,
        ) as zipf:
            for idx, entry in enumerate(sequence_entries, start=1):
                obj_file = str(entry["obj_file"])
                zipf.write(sequence_dir / obj_file, arcname=obj_file)
                if idx % report_every == 0 or idx == total_sequence_entries:
                    _check_cancel(should_abort)
                    pack_ratio = idx / float(max(1, total_sequence_entries))
                    pack_percent = min(99, 97 + int(round(pack_ratio * 2.0)))
                    _report_progress(
                        progress_callback,
                        stage="video_packaging",
                        percent=pack_percent,
                        message="packaging frame models",
                        total_frames=total_frames,
                        processed_frames=total_frames,
                        detected_frames=reconstructed_frames,
                        interpolated_frames=interpolated_frames,
                    )

        _report_progress(
            progress_callback,
            stage="video_metadata",
            percent=99,
            message="writing video metadata",
            total_frames=total_frames,
            processed_frames=total_frames,
            detected_frames=reconstructed_frames,
            interpolated_frames=interpolated_frames,
        )
        metadata = {
            "video_path": video_path,
            "total_frames": total_frames,
            "reconstructed_frames": reconstructed_frames,
            "keyframe_index": best_index,
            "animation_file": animation_path.name,
            "sequence_zip_file": sequence_zip_path.name,
            "preview_dir": preview_dir.name,
            "entries": sequence_entries,
        }
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

        attention_summary = _build_attention_summary(
            attention_entries,
            fps=fps,
            detected_frames=reconstructed_frames,
            interpolated_frames=interpolated_frames,
            scenario=scenario_key,
            rapid_turn_events=rapid_turn_events,
        )
        attention_metadata = {
            "scenario": scenario_key,
            "fps": round(float(fps), 4),
            "summary": attention_summary,
            "entries": attention_entries,
        }
        attention_metadata_path.write_text(json.dumps(attention_metadata, ensure_ascii=False, indent=2), encoding="utf-8")

        _check_cancel(should_abort)
        _report_progress(
            progress_callback,
            stage="video_done",
            percent=100,
            message="video reconstruction completed",
            total_frames=total_frames,
            processed_frames=total_frames,
            detected_frames=reconstructed_frames,
            interpolated_frames=interpolated_frames,
        )

        return {
            "model_path": str(model_path),
            "preview_path": str(preview_path),
            "sequence_zip_path": str(sequence_zip_path),
            "animation_path": str(animation_path),
            "metadata_path": str(metadata_path),
            "attention_metadata_path": str(attention_metadata_path),
            "keyframe_index": best_index,
            "log_text": "Video reconstructed with head-pose attention analysis. total_frames={}, detected_frames={}, interpolated_frames={}, rapid_turn_events={}".format(
                total_frames,
                reconstructed_frames,
                interpolated_frames,
                rapid_turn_events,
            ),
        }

_runner_local = threading.local()


def _get_runner() -> ThreeDDFARunner:
    runner = getattr(_runner_local, "runner", None)
    if runner is None:
        runner = ThreeDDFARunner()
        _runner_local.runner = runner
    return runner


def analyze_attention_frame(
    image_bytes: bytes,
    *,
    scenario: str = "classroom",
    mode: str = "single",
) -> Dict[str, object]:
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError("Failed to decode image bytes")
    return _get_runner()._analyze_faces(frame, scenario=scenario, mode=mode)


def run_reconstruction(
    media_type: str,
    media_path: str,
    output_dir: Path,
    stem: str,
    progress_callback: ProgressCallback = None,
    should_abort: CancelCheck = None,
    attention_scenario: str = "classroom",
) -> Dict[str, Optional[str]]:
    runner = _get_runner()
    if media_type == "photo":
        return runner.reconstruct_photo(
            media_path,
            output_dir,
            stem,
            progress_callback=progress_callback,
            should_abort=should_abort,
            attention_scenario=attention_scenario,
        )
    if media_type == "video":
        return runner.reconstruct_video(
            media_path,
            output_dir,
            stem,
            progress_callback=progress_callback,
            should_abort=should_abort,
            attention_scenario=attention_scenario,
        )
    raise RuntimeError("Unknown media type: {}".format(media_type))
