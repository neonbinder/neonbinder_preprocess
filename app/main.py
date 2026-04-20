"""FastAPI entrypoint for the neonbinder-preprocess service.

Slice 1: `/health` + `/process` with orient→rotate→classify pipeline.
Slice 2a: optional `precropped` multipart field, crop cascade.
Slice 2b: SAM added to the cascade; text-count + classify-error gates
          applied uniformly across every crop strategy via the wrapper
          in `app.cropper`. Main.py is now a thin layer: auth + upload
          validation → cropper.crop() → response packaging.
"""

from __future__ import annotations

import base64
import hmac
import logging
import os
from typing import Annotated

from fastapi import FastAPI, File, Header, HTTPException, UploadFile, status
from pydantic import BaseModel

from app import cropper
from app.classify import ClassifyError

logger = logging.getLogger(__name__)

app = FastAPI(title="neonbinder-preprocess", version="0.3.0")

INTERNAL_API_KEY_ENV = "INTERNAL_API_KEY"
MAX_IMAGE_BYTES = 32 * 1024 * 1024
ALLOWED_CONTENT_TYPES = frozenset({"image/jpeg", "image/png", "image/webp"})


class ProcessResponse(BaseModel):
    """Response body for POST /process.

    `players` is the canonical list of every player visible on the card
    (one entry for single-player cards, many for leaders/combo/dual-
    rookie/team set cards, empty when unidentifiable). `player` is a
    back-compat convenience: first entry or null.

    `rotation_degrees` is the CCW rotation that was applied to the chosen
    crop before classification; clients that store the corrected image
    should apply the same rotation to keep their copy aligned with what
    the model actually saw.

    `cropped_source` tells the client which stage of the crop cascade
    won. When it is `"precropped"` the client's upload was used as-is and
    `cropped_image_b64` will be null (the client already has the bytes
    on disk). For every other source, the server produced new bytes and
    returns them base64-encoded in `cropped_image_b64`.
    """

    players: list[str]
    player: str | None
    team: str | None
    card_number: str | None
    side: str
    rotation_degrees: int
    orient_confidence: float
    text_count: int
    cropped_source: str
    cropped_image_b64: str | None


def _verify_internal_key(x_internal_key: str | None) -> None:
    expected = os.environ.get(INTERNAL_API_KEY_ENV)
    if not expected:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="internal api key not configured",
        )
    if not x_internal_key or not hmac.compare_digest(x_internal_key, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid internal key",
        )


def _validate_content_type(content_type: str | None, *, field: str) -> None:
    if (content_type or "").split(";")[0].strip() not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"unsupported content-type for {field}: {content_type}",
        )


async def _read_upload(upload: UploadFile, *, field: str) -> bytes:
    _validate_content_type(upload.content_type, field=field)
    data = await upload.read()
    if not data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"empty {field}",
        )
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"{field} exceeds max size of {MAX_IMAGE_BYTES} bytes",
        )
    return data


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/process", response_model=ProcessResponse)
async def process(
    image: Annotated[UploadFile, File()],
    precropped: Annotated[UploadFile | None, File()] = None,
    x_internal_key: Annotated[str | None, Header()] = None,
) -> ProcessResponse:
    _verify_internal_key(x_internal_key)

    image_bytes = await _read_upload(image, field="image")
    precropped_bytes: bytes | None = None
    if precropped is not None:
        precropped_bytes = await _read_upload(precropped, field="precropped")

    # The cascade handles orient + rotate + classify internally so every
    # crop strategy is evaluated through the same quality gates. Upstream
    # failures (Vision / Anthropic) surface as exceptions we translate to
    # 502 here.
    try:
        result = cropper.crop(image_bytes=image_bytes, precropped_bytes=precropped_bytes)
    except ClassifyError:
        logger.exception("classify failed to parse after retry")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="classify response unparseable",
        ) from None
    except Exception:
        logger.exception("cascade failed")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="preprocess pipeline upstream failure",
        ) from None

    cropped_image_b64: str | None = None
    if result.returned_bytes_differ:
        cropped_image_b64 = base64.b64encode(result.image_bytes).decode("ascii")

    return ProcessResponse(
        players=list(result.classification.players),
        player=result.classification.player,
        team=result.classification.team,
        card_number=result.classification.card_number,
        side=result.classification.side,
        rotation_degrees=result.orientation.rotation_degrees,
        orient_confidence=result.orientation.confidence,
        text_count=result.orientation.text_count,
        cropped_source=result.source,
        cropped_image_b64=cropped_image_b64,
    )
