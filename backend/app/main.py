"""
IndicSent-M: FastAPI Main Application
=====================================
Unified entry point for multimodal sentiment analysis backend.
"""

import logging
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError

from app.core.config import settings
from app.core.errors import (
    AppException,
    app_exception_handler,
    validation_exception_handler,
    generic_exception_handler
)
from app.services.predictor_service import PredictorService, get_predictor_service
from app.api.routes import health, models, prediction, ocr, explanation, report, suggestions

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("indicsent.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Initializes models once during startup and cleans up on shutdown.
    """
    logger.info("Starting IndicSent-M Backend...")
    service = get_predictor_service()
    service.initialize()
    logger.info("IndicSent-M Backend is ready to serve requests.")
    yield
    logger.info("Shutting down IndicSent-M Backend...")
    service.cleanup()
    logger.info("Backend shutdown complete.")


# Initialize FastAPI app
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description=settings.DESCRIPTION,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Configure CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Register Exception Handlers
app.add_exception_handler(AppException, app_exception_handler)
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(Exception, generic_exception_handler)

# Register API Routers
app.include_router(health.router, prefix=settings.API_PREFIX)
app.include_router(models.router, prefix=settings.API_PREFIX)
app.include_router(prediction.router, prefix=settings.API_PREFIX)
app.include_router(ocr.router, prefix=settings.API_PREFIX)
app.include_router(explanation.router, prefix=settings.API_PREFIX)
app.include_router(report.router, prefix=settings.API_PREFIX)
app.include_router(suggestions.router, prefix=settings.API_PREFIX)


@app.get("/", tags=["Root"], summary="API Root", description="Returns basic API metadata and links to documentation.")
async def root():
    return {
        "name": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "status": "online",
        "docs_url": "/docs",
        "endpoints": {
            "health": f"{settings.API_PREFIX}/health",
            "models": f"{settings.API_PREFIX}/models",
            "predict_text": f"{settings.API_PREFIX}/predict/text",
            "predict_image": f"{settings.API_PREFIX}/predict/image",
            "predict_multimodal": f"{settings.API_PREFIX}/predict/multimodal",
            "ocr_extract": f"{settings.API_PREFIX}/ocr/extract",
            "explain": f"{settings.API_PREFIX}/explain"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)
