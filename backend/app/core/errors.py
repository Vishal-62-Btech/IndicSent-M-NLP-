"""
IndicSent-M: Custom Error Classes and Handlers
"""

from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError


class AppException(Exception):
    def __init__(self, status_code: int, message: str, code: str = "ERROR"):
        self.status_code = status_code
        self.message = message
        self.code = code
        super().__init__(message)


class BadRequestException(AppException):
    def __init__(self, message: str):
        super().__init__(status_code=status.HTTP_400_BAD_REQUEST, message=message, code="BAD_REQUEST")


class FileTooLargeException(AppException):
    def __init__(self, message: str = "Uploaded file exceeds maximum allowed size (10 MB)."):
        status_code = getattr(status, "HTTP_413_CONTENT_TOO_LARGE", status.HTTP_413_REQUEST_ENTITY_TOO_LARGE)
        super().__init__(status_code=status_code, message=message, code="FILE_TOO_LARGE")


class UnsupportedMediaTypeException(AppException):
    def __init__(self, message: str = "Unsupported image format or MIME type."):
        super().__init__(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, message=message, code="UNSUPPORTED_MEDIA_TYPE")


class ModelNotFoundException(AppException):
    def __init__(self, message: str):
        super().__init__(status_code=status.HTTP_400_BAD_REQUEST, message=message, code="MODEL_NOT_FOUND")


class ModelUnavailableException(AppException):
    def __init__(self, message: str):
        super().__init__(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, message=message, code="MODEL_UNAVAILABLE")


class InferenceException(AppException):
    def __init__(self, message: str):
        super().__init__(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, message=message, code="INFERENCE_FAILED")


async def app_exception_handler(request: Request, exc: AppException) -> JSONResponse:
    """Formats custom application exceptions into clean JSON without exposing stack traces."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": {
                "code": exc.code,
                "message": exc.message
            }
        }
    )


async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Formats Pydantic/FastAPI validation errors into clean HTTP 422 responses."""
    details = []
    for err in exc.errors():
        loc = " -> ".join([str(l) for l in err.get("loc", [])])
        msg = err.get("msg", "Validation error")
        details.append(f"{loc}: {msg}")
    status_code = getattr(status, "HTTP_422_UNPROCESSABLE_CONTENT", status.HTTP_422_UNPROCESSABLE_ENTITY)
    return JSONResponse(
        status_code=status_code,
        content={
            "success": False,
            "error": {
                "code": "VALIDATION_ERROR",
                "message": "; ".join(details)
            }
        }
    )


async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Fallback handler for unexpected server errors (HTTP 500) hiding internal system details."""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error": {
                "code": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected error occurred during prediction processing."
            }
        }
    )
