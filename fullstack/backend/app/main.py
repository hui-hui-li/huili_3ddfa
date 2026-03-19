from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.admin import router as admin_router
from app.api.attention import router as attention_router
from app.api.auth import router as auth_router
from app.api.media import router as media_router
from app.api.reconstructions import router as recon_router
from app.core.config import settings
from app.core.database import init_db
from app.services.job_queue import recover_orphaned_jobs, shutdown_job_executor
from app.services.storage import ensure_storage_dirs


app = FastAPI(title=settings.app_name)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup() -> None:
    ensure_storage_dirs()
    init_db()
    recover_orphaned_jobs()


@app.on_event("shutdown")
def on_shutdown() -> None:
    shutdown_job_executor()


app.include_router(auth_router, prefix=settings.api_prefix)
app.include_router(admin_router, prefix=settings.api_prefix)
app.include_router(media_router, prefix=settings.api_prefix)
app.include_router(recon_router, prefix=settings.api_prefix)
app.include_router(attention_router, prefix=settings.api_prefix)


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


if settings.frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(settings.frontend_dir), html=True), name="frontend")


@app.get("/")
def index():
    index_file = settings.frontend_dir / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "Frontend not found"}
