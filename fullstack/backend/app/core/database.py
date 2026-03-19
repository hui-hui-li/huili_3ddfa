from __future__ import annotations

from collections.abc import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.orm import declarative_base, sessionmaker

from .config import settings


connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}
engine = create_engine(settings.database_url, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db() -> Generator:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    from . import models  # noqa: F401

    Base.metadata.create_all(bind=engine)
    ensure_schema_columns()


def _ensure_columns(table_name: str, column_defs: dict) -> None:
    if not settings.database_url.startswith("sqlite"):
        return

    with engine.begin() as conn:
        existing = {
            row[1]
            for row in conn.execute(text("PRAGMA table_info({})".format(table_name))).fetchall()
        }
        for column_name, column_ddl in column_defs.items():
            if column_name not in existing:
                conn.execute(
                    text(
                        "ALTER TABLE {} ADD COLUMN {} {}".format(
                            table_name, column_name, column_ddl
                        )
                    )
                )


def _drop_columns(table_name: str, column_names: list[str]) -> None:
    if not settings.database_url.startswith("sqlite"):
        return

    with engine.begin() as conn:
        existing = {
            row[1]
            for row in conn.execute(text("PRAGMA table_info({})".format(table_name))).fetchall()
        }
        for column_name in column_names:
            if column_name in existing:
                conn.execute(text("ALTER TABLE {} DROP COLUMN {}".format(table_name, column_name)))
                existing.remove(column_name)


def _drop_table_if_exists(table_name: str) -> None:
    if not settings.database_url.startswith("sqlite"):
        return
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS {}".format(table_name)))


def _backfill_reconstruction_task_names() -> None:
    if not settings.database_url.startswith("sqlite"):
        return

    with engine.begin() as conn:
        conn.execute(
            text(
                """
                UPDATE reconstruction_jobs
                SET task_name = COALESCE(
                    NULLIF(task_name, ''),
                    (
                        SELECT media_assets.original_filename
                        FROM media_assets
                        WHERE media_assets.id = reconstruction_jobs.media_id
                    ),
                    'media-' || reconstruction_jobs.media_id
                )
                WHERE task_name IS NULL OR TRIM(task_name) = ''
                """
            )
        )


def _backfill_media_attention_scenarios() -> None:
    if not settings.database_url.startswith("sqlite"):
        return

    with engine.begin() as conn:
        conn.execute(
            text(
                """
                UPDATE media_assets
                SET default_attention_scenario = COALESCE(
                    (
                        SELECT CASE
                            WHEN LOWER(TRIM(reconstruction_jobs.attention_scenario)) IN ('exam', 'driving')
                                THEN LOWER(TRIM(reconstruction_jobs.attention_scenario))
                            ELSE NULL
                        END
                        FROM reconstruction_jobs
                        WHERE reconstruction_jobs.media_id = media_assets.id
                        ORDER BY reconstruction_jobs.created_at ASC, reconstruction_jobs.id ASC
                        LIMIT 1
                    ),
                    CASE
                        WHEN media_assets.original_filename LIKE '%驾驶%' OR LOWER(media_assets.original_filename) LIKE '%driving%' THEN 'driving'
                        WHEN media_assets.original_filename LIKE '%考试%' OR LOWER(media_assets.original_filename) LIKE '%exam%' THEN 'exam'
                        WHEN media_assets.original_filename LIKE '%课堂%' OR media_assets.original_filename LIKE '%上课%' OR LOWER(media_assets.original_filename) LIKE '%classroom%' THEN 'classroom'
                        ELSE NULL
                    END,
                    (
                        SELECT CASE
                            WHEN LOWER(TRIM(reconstruction_jobs.attention_scenario)) IN ('classroom', 'exam', 'driving')
                                THEN LOWER(TRIM(reconstruction_jobs.attention_scenario))
                            ELSE NULL
                        END
                        FROM reconstruction_jobs
                        WHERE reconstruction_jobs.media_id = media_assets.id
                        ORDER BY reconstruction_jobs.created_at ASC, reconstruction_jobs.id ASC
                        LIMIT 1
                    ),
                    'classroom'
                )
                WHERE default_attention_scenario IS NULL
                    OR TRIM(default_attention_scenario) = ''
                    OR LOWER(TRIM(default_attention_scenario)) NOT IN ('classroom', 'exam', 'driving')
                    OR (
                        LOWER(TRIM(default_attention_scenario)) = 'classroom'
                        AND (
                            media_assets.original_filename LIKE '%驾驶%'
                            OR LOWER(media_assets.original_filename) LIKE '%driving%'
                            OR media_assets.original_filename LIKE '%考试%'
                            OR LOWER(media_assets.original_filename) LIKE '%exam%'
                            OR EXISTS(
                                SELECT 1
                                FROM reconstruction_jobs
                                WHERE reconstruction_jobs.media_id = media_assets.id
                                    AND LOWER(TRIM(reconstruction_jobs.attention_scenario)) IN ('exam', 'driving')
                            )
                        )
                    )
                """
            )
        )


def ensure_schema_columns() -> None:
    _ensure_columns(
        "users",
        {
            "is_admin": "INTEGER NOT NULL DEFAULT 0",
            "is_banned": "INTEGER NOT NULL DEFAULT 0",
            "ban_reason": "TEXT",
            "must_reset_password": "INTEGER NOT NULL DEFAULT 0",
        },
    )
    _ensure_columns(
        "reconstruction_jobs",
        {
            "task_name": "TEXT",
            "attention_scenario": "TEXT NOT NULL DEFAULT 'classroom'",
            "output_sequence_zip_path": "TEXT",
            "output_animation_path": "TEXT",
            "output_metadata_path": "TEXT",
            "output_attention_metadata_path": "TEXT",
            "keyframe_index": "INTEGER",
            "progress_percent": "INTEGER NOT NULL DEFAULT 0",
            "progress_stage": "TEXT",
            "progress_message": "TEXT",
            "progress_updated_at": "DATETIME",
            "total_frames": "INTEGER",
            "processed_frames": "INTEGER",
        },
    )
    _ensure_columns(
        "media_assets",
        {
            "default_attention_scenario": "TEXT NOT NULL DEFAULT 'classroom'",
        },
    )
    _drop_columns(
        "reconstruction_jobs",
        [
            "avatar_profile_id",
            "output_avatar_animation_path",
            "output_avatar_preview_path",
            "output_avatar_metadata_path",
            "output_avatar_frames_dir",
        ],
    )
    _drop_table_if_exists("avatar_profiles")
    _backfill_reconstruction_task_names()
    _backfill_media_attention_scenarios()
