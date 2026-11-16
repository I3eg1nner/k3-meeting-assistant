from __future__ import annotations

import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .api import ApiContext, build_router
from .config import Settings
from .db import Database
from .inference import InferenceRegistry
from .pipeline import PipelineRunner


settings = Settings.from_env()
settings.ensure_directories()
db = Database(settings.db_path)
inference = InferenceRegistry(settings)
pipeline = PipelineRunner(settings, db, inference)
context = ApiContext(settings, db, pipeline)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    db.initialize()
    stale_jobs = db.all(
        "SELECT id FROM jobs WHERE status IN ('QUEUED','RUNNING') ORDER BY created_at"
    )
    if stale_jobs:
        def recover() -> None:
            for job in stale_jobs:
                pipeline.run(job["id"])

        threading.Thread(target=recover, name="meeting-job-recovery", daemon=True).start()
    yield
    inference.model_manager.stop()
    inference.vlm_manager.stop()


app = FastAPI(
    title="K3 会议智记",
    description="Evidence-first meeting assistant for SpacemiT K3",
    version="0.1.0",
    lifespan=lifespan,
)
app.include_router(build_router(context))

assets_dir = settings.static_dir / "assets"
if assets_dir.exists():
    app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")


@app.get("/{path:path}", include_in_schema=False)
def frontend(path: str):
    requested = settings.static_dir / path
    if path and requested.is_file() and settings.static_dir in requested.resolve().parents:
        return FileResponse(requested)
    index = settings.static_dir / "index.html"
    if index.exists():
        return FileResponse(index)
    return JSONResponse(
        {"message": "前端尚未构建，请运行 npm run build:k3", "api_docs": "/docs"},
        status_code=503,
    )
