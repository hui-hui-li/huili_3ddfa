from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from .database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(64), unique=True, index=True, nullable=False)
    email = Column(String(255), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    is_admin = Column(Boolean, default=False, nullable=False)
    is_banned = Column(Boolean, default=False, nullable=False)
    ban_reason = Column(Text, nullable=True)
    must_reset_password = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    media_items = relationship("MediaAsset", back_populates="owner", cascade="all, delete-orphan")
    jobs = relationship("ReconstructionJob", back_populates="owner", cascade="all, delete-orphan")
    admin_action_logs = relationship(
        "AdminAuditLog",
        back_populates="admin_user",
        foreign_keys="AdminAuditLog.admin_user_id",
        cascade="all, delete-orphan",
    )
    admin_target_logs = relationship(
        "AdminAuditLog",
        back_populates="target_user",
        foreign_keys="AdminAuditLog.target_user_id",
    )


class MediaAsset(Base):
    __tablename__ = "media_assets"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)

    media_type = Column(String(16), nullable=False, index=True)  # photo | video
    original_filename = Column(String(255), nullable=False)
    stored_filename = Column(String(255), nullable=False)
    stored_path = Column(String(1024), nullable=False)
    mime_type = Column(String(255), nullable=True)
    file_size = Column(Integer, nullable=False)
    default_attention_scenario = Column(String(24), nullable=False, default="classroom")  # classroom | exam | driving

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    owner = relationship("User", back_populates="media_items")
    jobs = relationship("ReconstructionJob", back_populates="media", cascade="all, delete-orphan")


class ReconstructionJob(Base):
    __tablename__ = "reconstruction_jobs"

    id = Column(String(36), primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    media_id = Column(Integer, ForeignKey("media_assets.id"), nullable=False, index=True)
    task_name = Column(String(255), nullable=True)

    status = Column(String(24), nullable=False, default="queued", index=True)  # queued | running | completed | failed | cancelled
    attention_scenario = Column(String(24), nullable=False, default="classroom")  # classroom | exam | driving
    output_model_path = Column(String(1024), nullable=True)
    output_preview_path = Column(String(1024), nullable=True)
    output_sequence_zip_path = Column(String(1024), nullable=True)
    output_animation_path = Column(String(1024), nullable=True)
    output_metadata_path = Column(String(1024), nullable=True)
    output_attention_metadata_path = Column(String(1024), nullable=True)
    progress_percent = Column(Integer, nullable=False, default=0)
    progress_stage = Column(String(64), nullable=True)
    progress_message = Column(Text, nullable=True)
    progress_updated_at = Column(DateTime, nullable=True)
    total_frames = Column(Integer, nullable=True)
    processed_frames = Column(Integer, nullable=True)
    keyframe_index = Column(Integer, nullable=True)
    log_text = Column(Text, nullable=True)
    error_message = Column(Text, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    started_at = Column(DateTime, nullable=True)
    finished_at = Column(DateTime, nullable=True)

    owner = relationship("User", back_populates="jobs")
    media = relationship("MediaAsset", back_populates="jobs")


class AdminAuditLog(Base):
    __tablename__ = "admin_audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    admin_user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    target_user_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)
    action = Column(String(64), nullable=False, index=True)
    detail_json = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    admin_user = relationship("User", back_populates="admin_action_logs", foreign_keys=[admin_user_id])
    target_user = relationship("User", back_populates="admin_target_logs", foreign_keys=[target_user_id])
