from __future__ import annotations

from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, EmailStr, Field


class RegisterRequest(BaseModel):
    username: str = Field(min_length=3, max_length=64)
    email: EmailStr
    password: str = Field(min_length=6, max_length=128)


class LoginRequest(BaseModel):
    username_or_email: str
    password: str


class ChangePasswordRequest(BaseModel):
    old_password: str
    new_password: str = Field(min_length=6, max_length=128)


class DeleteAccountRequest(BaseModel):
    password: str = Field(min_length=1, max_length=128)


class UpdateProfileRequest(BaseModel):
    username: Optional[str] = Field(default=None, min_length=3, max_length=64)
    email: Optional[EmailStr] = None


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserResponse(BaseModel):
    id: int
    username: str
    email: EmailStr
    is_admin: bool
    is_banned: bool
    ban_reason: Optional[str] = None
    must_reset_password: bool
    created_at: datetime


class AdminUserResponse(BaseModel):
    id: int
    username: str
    email: EmailStr
    is_admin: bool
    is_banned: bool
    ban_reason: Optional[str] = None
    must_reset_password: bool
    created_at: datetime


class BanUserRequest(BaseModel):
    reason: Optional[str] = None


class SetAdminRequest(BaseModel):
    is_admin: bool


class ResetPasswordRequest(BaseModel):
    new_password: Optional[str] = Field(default=None, min_length=6, max_length=128)


class ResetPasswordResponse(BaseModel):
    user_id: int
    temporary_password: str


class AdminAuditLogResponse(BaseModel):
    id: int
    admin_user_id: int
    admin_username: Optional[str] = None
    target_user_id: Optional[int] = None
    target_username: Optional[str] = None
    action: str
    detail_json: Optional[str] = None
    created_at: datetime


class MediaResponse(BaseModel):
    id: int
    media_type: Literal["photo", "video"]
    original_filename: str
    file_size: int
    mime_type: Optional[str] = None
    default_attention_scenario: Literal["classroom", "exam", "driving"] = "classroom"
    created_at: datetime


class MediaBatchDeleteRequest(BaseModel):
    media_ids: List[int]


class MediaBatchDeleteResponse(BaseModel):
    deleted_ids: List[int] = Field(default_factory=list)
    blocked_ids: List[int] = Field(default_factory=list)
    missing_ids: List[int] = Field(default_factory=list)
    detail: str


class ReconstructionCreateRequest(BaseModel):
    media_id: int
    attention_scenario: Optional[Literal["classroom", "exam", "driving"]] = None


class ReconstructionBatchCreateItem(BaseModel):
    media_id: int
    job_id: str
    attention_scenario: Literal["classroom", "exam", "driving"] = "classroom"


class ReconstructionBatchCreateRequest(BaseModel):
    media_ids: List[int]
    attention_scenario: Optional[Literal["classroom", "exam", "driving"]] = None


class ReconstructionBatchCreateResponse(BaseModel):
    created: List[ReconstructionBatchCreateItem] = Field(default_factory=list)
    blocked_media_ids: List[int] = Field(default_factory=list)
    missing_media_ids: List[int] = Field(default_factory=list)
    detail: str


class ReconstructionBatchDeleteRequest(BaseModel):
    job_ids: List[str]


class ReconstructionBatchDeleteResponse(BaseModel):
    deleted_ids: List[str] = Field(default_factory=list)
    blocked_ids: List[str] = Field(default_factory=list)
    missing_ids: List[str] = Field(default_factory=list)
    detail: str


class ReconstructionBatchCancelRequest(BaseModel):
    job_ids: List[str]


class ReconstructionBatchCancelResponse(BaseModel):
    requested_ids: List[str] = Field(default_factory=list)
    cancelled_ids: List[str] = Field(default_factory=list)
    already_terminal_ids: List[str] = Field(default_factory=list)
    missing_ids: List[str] = Field(default_factory=list)
    detail: str


class ReconstructionResponse(BaseModel):
    id: str
    media_id: int
    task_name: str
    media_type: Literal["photo", "video"]
    status: str
    attention_scenario: Literal["classroom", "exam", "driving"] = "classroom"
    output_model_path: Optional[str] = None
    output_preview_path: Optional[str] = None
    output_sequence_zip_path: Optional[str] = None
    output_animation_path: Optional[str] = None
    output_metadata_path: Optional[str] = None
    output_attention_metadata_path: Optional[str] = None
    progress_percent: int = 0
    progress_stage: Optional[str] = None
    progress_message: Optional[str] = None
    total_frames: Optional[int] = None
    processed_frames: Optional[int] = None
    keyframe_index: Optional[int] = None
    error_message: Optional[str] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None


class FramePreviewEntry(BaseModel):
    frame_index: int
    image_url: str


class FramePreviewListResponse(BaseModel):
    total: int
    page: int
    page_size: int
    entries: List[FramePreviewEntry]


class AttentionTimelineEntry(BaseModel):
    frame_index: int
    yaw: float
    pitch: float
    roll: float
    attention_score: float
    source: str
    detection_score: float
    head_down: bool
    side_view: bool
    tilted: bool
    distracted: bool
    rapid_turn: bool


class AttentionTimelineResponse(BaseModel):
    total: int
    page: int
    page_size: int
    entries: List[AttentionTimelineEntry]


class AttentionCurvePoint(BaseModel):
    frame_index: int
    attention_score: float
    moving_avg_score: float
    distracted: bool
    rapid_turn: bool


class AttentionCurveResponse(BaseModel):
    scenario: str
    total_frames: int
    sampled_points: int
    points: List[AttentionCurvePoint]


class AttentionSummaryResponse(BaseModel):
    scenario: str
    fps: float
    total_frames: int
    detected_frames: int
    interpolated_frames: int
    avg_attention: float
    min_attention: float
    max_attention: float
    low_attention_ratio: float
    head_down_ratio: float
    side_view_ratio: float
    rapid_turn_events: int
    longest_distracted_frames: int
    classroom_head_up_rate: float
    exam_focus_score: float
    driving_risk_score: float
    warnings: List[str]


class RealtimeAttentionFace(BaseModel):
    face_index: int
    yaw: float
    pitch: float
    roll: float
    attention_score: float
    bbox_x: Optional[float] = None
    bbox_y: Optional[float] = None
    bbox_w: Optional[float] = None
    bbox_h: Optional[float] = None
    head_down: bool
    side_view: bool
    tilted: bool
    distracted: bool


class RealtimeAttentionResponse(BaseModel):
    mode: Literal["single", "multi"]
    scenario: str
    timestamp: datetime
    face_count: int
    avg_attention: float
    classroom_head_up_rate: float
    faces: List[RealtimeAttentionFace]


class FaceSwapSourceOption(BaseModel):
    job_id: str
    task_name: str
    media_type: Literal["photo", "video"]
    attention_scenario: Literal["classroom", "exam", "driving"] = "classroom"
    created_at: datetime


class FaceSwapFrameResponse(BaseModel):
    mode: Literal["single", "multi"]
    scenario: Literal["classroom", "exam", "driving"]
    timestamp: datetime
    source_job_id: str
    selected_target_face_index: int = 0
    face_count: int
    avg_attention: float
    classroom_head_up_rate: float
    replaced: bool
    swapped_image_base64: str
    faces: List[RealtimeAttentionFace]
