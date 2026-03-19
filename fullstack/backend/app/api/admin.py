from __future__ import annotations

import secrets
import string
from typing import List

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session, joinedload

from app.core.database import get_db
from app.core.dependencies import get_current_admin_user
from app.core.models import AdminAuditLog, User
from app.core.schemas import (
    AdminAuditLogResponse,
    AdminUserResponse,
    BanUserRequest,
    ResetPasswordRequest,
    ResetPasswordResponse,
    SetAdminRequest,
)
from app.core.security import hash_password
from app.services.audit import log_admin_action


router = APIRouter(prefix="/admin", tags=["admin"])


def _serialize_user(user: User) -> AdminUserResponse:
    return AdminUserResponse(
        id=user.id,
        username=user.username,
        email=user.email,
        is_admin=user.is_admin,
        is_banned=user.is_banned,
        ban_reason=user.ban_reason,
        must_reset_password=user.must_reset_password,
        created_at=user.created_at,
    )


def _serialize_log(log_entry: AdminAuditLog) -> AdminAuditLogResponse:
    return AdminAuditLogResponse(
        id=log_entry.id,
        admin_user_id=log_entry.admin_user_id,
        admin_username=log_entry.admin_user.username if log_entry.admin_user else None,
        target_user_id=log_entry.target_user_id,
        target_username=log_entry.target_user.username if log_entry.target_user else None,
        action=log_entry.action,
        detail_json=log_entry.detail_json,
        created_at=log_entry.created_at,
    )


def _generate_temporary_password(length: int = 12) -> str:
    alphabet = string.ascii_letters + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(length))


@router.get("/users", response_model=List[AdminUserResponse])
def list_users(
    db: Session = Depends(get_db),
    _: User = Depends(get_current_admin_user),
) -> List[AdminUserResponse]:
    users = db.query(User).order_by(User.id.asc()).all()
    return [_serialize_user(user) for user in users]


@router.get("/audit-logs", response_model=List[AdminAuditLogResponse])
def list_audit_logs(
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
    _: User = Depends(get_current_admin_user),
) -> List[AdminAuditLogResponse]:
    logs = (
        db.query(AdminAuditLog)
        .options(
            joinedload(AdminAuditLog.admin_user),
            joinedload(AdminAuditLog.target_user),
        )
        .order_by(AdminAuditLog.created_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )
    return [_serialize_log(log_entry) for log_entry in logs]


@router.post("/users/{user_id}/ban", response_model=AdminUserResponse)
def ban_user(
    user_id: int,
    payload: BanUserRequest,
    db: Session = Depends(get_db),
    current_admin: User = Depends(get_current_admin_user),
) -> AdminUserResponse:
    target = db.query(User).filter(User.id == user_id).first()
    if target is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    if target.id == current_admin.id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="You cannot ban yourself")

    reason = payload.reason or "Banned by admin"
    target.is_banned = True
    target.ban_reason = reason
    db.commit()
    db.refresh(target)

    log_admin_action(
        db,
        admin_user_id=current_admin.id,
        action="ban_user",
        target_user_id=target.id,
        detail={"reason": reason},
    )
    return _serialize_user(target)


@router.post("/users/{user_id}/unban", response_model=AdminUserResponse)
def unban_user(
    user_id: int,
    db: Session = Depends(get_db),
    current_admin: User = Depends(get_current_admin_user),
) -> AdminUserResponse:
    target = db.query(User).filter(User.id == user_id).first()
    if target is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    target.is_banned = False
    target.ban_reason = None
    db.commit()
    db.refresh(target)

    log_admin_action(
        db,
        admin_user_id=current_admin.id,
        action="unban_user",
        target_user_id=target.id,
    )
    return _serialize_user(target)


@router.post("/users/{user_id}/set-admin", response_model=AdminUserResponse)
def set_admin_status(
    user_id: int,
    payload: SetAdminRequest,
    db: Session = Depends(get_db),
    current_admin: User = Depends(get_current_admin_user),
) -> AdminUserResponse:
    target = db.query(User).filter(User.id == user_id).first()
    if target is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    if not payload.is_admin and target.id == current_admin.id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="You cannot revoke your own admin role")

    if not payload.is_admin and target.is_admin:
        admin_count = db.query(User).filter(User.is_admin.is_(True)).count()
        if admin_count <= 1:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="At least one admin is required")

    target.is_admin = payload.is_admin
    db.commit()
    db.refresh(target)

    log_admin_action(
        db,
        admin_user_id=current_admin.id,
        action="set_admin",
        target_user_id=target.id,
        detail={"is_admin": payload.is_admin},
    )
    return _serialize_user(target)


@router.post("/users/{user_id}/reset-password", response_model=ResetPasswordResponse)
def reset_user_password(
    user_id: int,
    payload: ResetPasswordRequest,
    db: Session = Depends(get_db),
    current_admin: User = Depends(get_current_admin_user),
) -> ResetPasswordResponse:
    target = db.query(User).filter(User.id == user_id).first()
    if target is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    temporary_password = payload.new_password or _generate_temporary_password()
    target.password_hash = hash_password(temporary_password)
    target.must_reset_password = True
    db.commit()

    log_admin_action(
        db,
        admin_user_id=current_admin.id,
        action="reset_password",
        target_user_id=target.id,
        detail={"custom_password": bool(payload.new_password)},
    )
    return ResetPasswordResponse(user_id=target.id, temporary_password=temporary_password)
