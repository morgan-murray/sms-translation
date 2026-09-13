from __future__ import annotations

import os
from pathlib import Path

from fastapi.middleware.cors import CORSMiddleware

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from app.adapters import ModelUnavailableError, TranslationExecutionError
from app.domain import TranslationRequest, TranslationResponse
from app.security import authenticated
from app.service import TranslationService


STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(
    title="Private translation service",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)
app.state.service = TranslationService()


@app.middleware("http")
async def private_service(request: Request, call_next):
    if request.url.path != "/healthz" and not authenticated(request.headers.get("authorization")):
        return Response(
            status_code=401,
            headers={"WWW-Authenticate": 'Basic realm="Private translation"'},
        )
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store"
    response.headers["Pragma"] = "no-cache"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; script-src 'self'; style-src 'self'; "
        "img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'"
    )
    return response


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.get("/", include_in_schema=False)
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/api/v1/translate", response_model=TranslationResponse)
async def translate(payload: TranslationRequest):
    try:
        return await app.state.service.translate(payload)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ModelUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except TranslationExecutionError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.exception_handler(Exception)
async def unhandled_error(_: Request, __: Exception):
    return JSONResponse(status_code=500, content={"detail": "internal translation service error"})


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# Register last so preflight and error responses receive CORS headers.
allowed_origins = [value.strip() for value in os.environ.get("CORS_ALLOWED_ORIGINS", "").split(",") if value.strip()]
if "*" in allowed_origins:
    raise ValueError("CORS_ALLOWED_ORIGINS must contain explicit origins")
if allowed_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["POST", "GET"],
        allow_headers=["Authorization", "Content-Type"],
    )
