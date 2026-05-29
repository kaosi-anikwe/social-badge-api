import json
import traceback
from typing import Any, cast

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from loguru import logger
from slowapi.errors import RateLimitExceeded
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.sanitizer import data_sanitizer
from app.schemas.response import ErrorResponse


async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    exc_type = type(exc).__name__

    if hasattr(exc, "statement") and hasattr(exc, "params"):
        sql_exc = cast(Any, exc)
        exc_msg = data_sanitizer.sanitize_sql_for_logging(
            sql_exc.statement,
            sql_exc.params,
        )
    elif hasattr(exc, "response") and hasattr(cast(Any, exc).response, "text"):
        try:
            exc_msg = data_sanitizer.sanitize_exception_for_logging(
                json.loads(cast(Any, exc).response.text)
            )
        except Exception:
            exc_msg = data_sanitizer.sanitize_exception_for_logging(str(exc))
    else:
        exc_msg = data_sanitizer.sanitize_exception_for_logging(exc)

    if isinstance(exc, IntegrityError):
        return JSONResponse(
            status_code=409,
            content=ErrorResponse(message="Database constraint violation").model_dump(),
        )

    if isinstance(exc, SQLAlchemyError):
        exc_orig = getattr(exc, "orig", None)

        logger.opt(exception=True).error(
            "SQLAlchemyError -> {} | sanitized={} | orig={}",
            exc_type,
            exc_msg,
            data_sanitizer.sanitize_exception_for_logging(exc_orig),
        )
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(message="A database error occurred").model_dump(),
        )

    if isinstance(exc, RequestValidationError):
        errors = exc.errors()
        if not errors:
            msg = "Validation Error"
        else:
            err = errors[0]
            loc_parts = [str(item) for item in err["loc"] if item != "body"]
            loc = " ".join(part.replace("_", " ").title() for part in loc_parts)
            msg = err["msg"]
            if msg.startswith("Value error, "):
                msg = msg.replace("Value error, ", "")
            elif ":" in msg:
                msg = msg.split(":")[-1].strip()
            if msg:
                msg = msg[0].lower() + msg[1:]
            msg = f"{loc}: {msg}" if loc else msg.capitalize()

        return JSONResponse(
            status_code=422,
            content=ErrorResponse(message=msg).model_dump(),
        )

    if isinstance(exc, RateLimitExceeded):
        return JSONResponse(
            status_code=429,
            content=ErrorResponse(message="Rate limit exceeded").model_dump(),
        )

    if isinstance(exc, StarletteHTTPException):
        msg = str(exc.detail)
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(message=msg).model_dump(),
        )

    tb = traceback.extract_tb(exc.__traceback__)
    if tb:
        tb_str = "".join(traceback.format_list(tb))
        location = data_sanitizer.sanitize_for_logging(f"\nTraceback:\n{tb_str}")
    else:
        location = "No traceback available"

    logger.critical(
        "Unhandled exception -> {}: {}\nLocation: {}",
        exc_type,
        exc_msg,
        location,
    )

    return JSONResponse(
        status_code=500,
        content=ErrorResponse(message="Internal Server Error").model_dump(),
    )


def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(RateLimitExceeded, global_exception_handler)
    app.add_exception_handler(StarletteHTTPException, global_exception_handler)
    app.add_exception_handler(RequestValidationError, global_exception_handler)
    app.add_exception_handler(Exception, global_exception_handler)
