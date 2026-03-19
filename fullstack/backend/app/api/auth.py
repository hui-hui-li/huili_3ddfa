from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import or_
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.dependencies import get_current_user
from app.core.models import AdminAuditLog, ReconstructionJob, User
from app.core.schemas import (
    ChangePasswordRequest,
    DeleteAccountRequest,
    LoginRequest,
    RegisterRequest,
    TokenResponse,
    UpdateProfileRequest,
    UserResponse,
)
from app.core.security import create_access_token, hash_password, verify_password
from app.services.storage import delete_user_storage


router = APIRouter(prefix="/auth", tags=["auth"])


def _serialize_user(user: User) -> UserResponse:
    return UserResponse(
        id=user.id,
        username=user.username,
        email=user.email,
        is_admin=user.is_admin,
        is_banned=user.is_banned,
        ban_reason=user.ban_reason,
        must_reset_password=user.must_reset_password,
        created_at=user.created_at,
    )


@router.post("/register", response_model=TokenResponse)
def register(payload: RegisterRequest, db: Session = Depends(get_db)) -> TokenResponse:
    username = payload.username.strip()
    email = str(payload.email).strip().lower()
    if len(username) < 3:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Username must be at least 3 characters")
    if any(ch.isspace() for ch in username):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Username cannot contain spaces")

    existing = (
        db.query(User)
        .filter(or_(User.username == username, User.email == email))
        .first()
    )
    if existing:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Username or email already exists")

    is_first_user = db.query(User.id).count() == 0
    user = User(
        username=username,
        email=email,
        password_hash=hash_password(payload.password),
        is_admin=is_first_user,
        is_banned=False,
        must_reset_password=False,
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    token = create_access_token(str(user.id))
    return TokenResponse(access_token=token)


@router.post("/login", response_model=TokenResponse)
def login(payload: LoginRequest, db: Session = Depends(get_db)) -> TokenResponse:
    identity = payload.username_or_email.strip()
    if not identity:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Username or email is required")

    user = (
        db.query(User)
        .filter(or_(User.username == identity, User.email == identity.lower()))
        .first()
    )

    if user is None or not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid username/email or password")
    if user.is_banned:
        reason = user.ban_reason or "No reason provided"
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="User is banned: {}".format(reason))

    token = create_access_token(str(user.id))
    return TokenResponse(access_token=token)


@router.get("/me", response_model=UserResponse)
def me(current_user: User = Depends(get_current_user)) -> UserResponse:
    return _serialize_user(current_user)


@router.patch("/profile", response_model=UserResponse)
def update_profile(
    payload: UpdateProfileRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> UserResponse:
    username = payload.username.strip() if payload.username is not None else None
    email = str(payload.email).strip().lower() if payload.email is not None else None

    if username is not None and len(username) < 3:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Username must be at least 3 characters")
    if username is not None and any(ch.isspace() for ch in username):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Username cannot contain spaces")

    if username is None and email is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No profile fields provided")

    changed = False

    if username is not None and username != current_user.username:
        username_exists = db.query(User.id).filter(User.id != current_user.id, User.username == username).first()
        if username_exists:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Username already exists")
        current_user.username = username
        changed = True

    if email is not None and email != current_user.email:
        email_exists = db.query(User.id).filter(User.id != current_user.id, User.email == email).first()
        if email_exists:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email already exists")
        current_user.email = email
        changed = True

    if changed:
        db.commit()
        db.refresh(current_user)

    return _serialize_user(current_user)


@router.post("/change-password")
def change_password(
    payload: ChangePasswordRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if not verify_password(payload.old_password, current_user.password_hash):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Old password is incorrect")
    if payload.old_password == payload.new_password:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="New password must be different")

    current_user.password_hash = hash_password(payload.new_password)
    current_user.must_reset_password = False
    db.commit()
    return {"detail": "password updated"}


@router.post("/delete-account")
def delete_account(
    payload: DeleteAccountRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if not verify_password(payload.password, current_user.password_hash):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Current password is incorrect")

    active_job = (
        db.query(ReconstructionJob.id)
        .filter(
            ReconstructionJob.user_id == current_user.id,
            ReconstructionJob.status.in_(("queued", "running")),
        )
        .first()
    )
    if active_job is not None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Please cancel active reconstruction jobs before deleting your account",
        )

    if current_user.is_admin:
        other_user_exists = db.query(User.id).filter(User.id != current_user.id).first() is not None
        other_admin_exists = (
            db.query(User.id)
            .filter(User.id != current_user.id, User.is_admin.is_(True))
            .first()
            is not None
        )
        if other_user_exists and not other_admin_exists:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot delete the last admin account while other users still exist",
            )

    user_id = current_user.id
    db.query(AdminAuditLog).filter(AdminAuditLog.target_user_id == user_id).update(
        {AdminAuditLog.target_user_id: None},
        synchronize_session=False,
    )
    db.delete(current_user)
    db.commit()
    delete_user_storage(user_id)
    return {"detail": "account deleted"}
