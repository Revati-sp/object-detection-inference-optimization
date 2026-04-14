"""
FastAPI application entry point.

Startup sequence:
1. Ensure upload/output/weights directories exist.
2. Register API routers.
3. Serve OpenAPI docs at /docs and /redoc.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes_detection import router as detection_router
from app.api.routes_eval import router as eval_router
from app.core.config import get_settings
from app.core.logging import logger

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Create required directories before the first request."""
    settings.ensure_directories()
    logger.info("Object Detection API starting up | version=%s", settings.APP_VERSION)
    logger.info("Upload dir:  %s", settings.UPLOAD_DIR)
    logger.info("Output dir:  %s", settings.OUTPUT_DIR)
    logger.info("Weights dir: %s", settings.WEIGHTS_DIR)
    yield
    logger.info("Object Detection API shutting down.")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "Real-time object detection API supporting YOLOv8 and YOLOv5 with "
        "PyTorch, TorchScript, and ONNX Runtime inference backends."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------
app.include_router(detection_router)
app.include_router(eval_router)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
@app.get("/health", tags=["Health"], summary="Service health check")
def health():
    return {
        "status": "ok",
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
    }


# ---------------------------------------------------------------------------
# Dev server entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info",
    )
