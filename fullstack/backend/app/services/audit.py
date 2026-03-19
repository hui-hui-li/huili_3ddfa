from __future__ import annotations

import json
from typing import Any, Dict, Optional

from sqlalchemy.orm import Session

from app.core.models import AdminAuditLog


def log_admin_action(
    db: Session,
    admin_user_id: int,
    action: str,
    target_user_id: Optional[int] = None,
    detail: Optional[Dict[str, Any]] = None,
) -> AdminAuditLog:
    log_entry = AdminAuditLog(
        admin_user_id=admin_user_id,
        target_user_id=target_user_id,
        action=action,
        detail_json=json.dumps(detail or {}, ensure_ascii=False),
    )
    db.add(log_entry)
    db.commit()
    db.refresh(log_entry)
    return log_entry
