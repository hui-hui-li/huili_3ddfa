from __future__ import annotations

import os
from pathlib import Path


class Settings:
    def __init__(self) -> None:
        backend_root = Path(__file__).resolve().parents[2]
        fullstack_root = backend_root.parent
        project_root = fullstack_root.parent

        self.backend_root = backend_root
        self.fullstack_root = fullstack_root
        self.project_root = project_root

        self.app_name = os.getenv("APP_NAME", "3DDFA Reconstruction Platform")
        self.api_prefix = os.getenv("API_PREFIX", "/api")

        self.jwt_secret = os.getenv("JWT_SECRET", "change-me-in-production")
        self.jwt_algorithm = os.getenv("JWT_ALGORITHM", "HS256")
        self.token_expire_minutes = int(os.getenv("TOKEN_EXPIRE_MINUTES", "1440"))

        db_path = backend_root / "storage" / "app.db"
        self.database_url = os.getenv("DATABASE_URL", f"sqlite:///{db_path}")

        self.storage_root = Path(os.getenv("STORAGE_ROOT", str(backend_root / "storage")))
        self.uploads_root = self.storage_root / "uploads"
        self.outputs_root = self.storage_root / "outputs"

        self.max_upload_size_mb = int(os.getenv("MAX_UPLOAD_SIZE_MB", "800"))
        self.video_frame_stride = int(os.getenv("VIDEO_FRAME_STRIDE", "10"))
        self.use_onnx = os.getenv("USE_ONNX", "true").lower() == "true"
        cpu_count = max(1, os.cpu_count() or 1)
        self.reconstruction_max_workers = max(
            1,
            min(int(os.getenv("RECONSTRUCTION_MAX_WORKERS", "2")), cpu_count),
        )
        self.inference_threads_per_worker = max(
            1,
            int(os.getenv("INFERENCE_THREADS_PER_WORKER", "1")),
        )

        self.photo_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        self.video_exts = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

        self.frontend_dir = fullstack_root / "frontend"


settings = Settings()
