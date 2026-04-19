"""FastAPI entrypoint for the neonbinder-preprocess service.

Slice 1 (current): `/health` + `/process`. `/process` accepts an already-
cropped card image, runs orient + classify, and returns structured card
fields plus the CCW rotation that was applied before classification.

Slice 2 (future): `/crop-and-process` adding SAM fallback for raw photos.
"""

from __future__ import annotations

import hmac
import logging
import os
from io import BytesIO
from typing import Annotated

from fastapi import FastAPI, File, Header, HTTPException, UploadFile, status
from PIL import Image
from pydantic import BaseModel

from app.classify import ClassifyError, classify_card
from app.orient import detect_orientation

logger = logging.getLogger(__name__)

app = FastAPI(title="neonbinder-preprocess", version="0.1.0")

INTERNAL_API_KEY_ENV = "INTERNAL_API_KEY"
MAX_IMAGE_BYTES = 32 * 1024 * 1024
ALLOWED_CONTENT_TYPES = frozenset({"image/jpeg", "image/png", "image/webp"})


class ProcessResponse(BaseModel):
    """Response body for POST /process.

    `rotation_degrees` is the CCW rotation that was applied to the image
    before classification; clients that store the corrected image should
    apply the same rotation to keep their copy aligned with what the model
    actually saw.
    """

    player: str | None
    team: str | None
    card_number: str | None
    side: str
    rotation_degrees: int
    orient_confidence: float
    text_count: int


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


def _validate_content_type(content_type: str | None) -> None:
    if (content_type or "").split(";")[0].strip() not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"unsupported content-type: {content_type}",
        )


def _rotate_image_bytes(image_bytes: bytes, degrees_ccw: int) -> bytes:
    if degrees_ccw % 360 == 0:
        return image_bytes
    with Image.open(BytesIO(image_bytes)) as img:
        fmt = img.format or "JPEG"
        rotated = img.rotate(degrees_ccw, expand=True)
        out = BytesIO()
        rotated.save(out, format=fmt)
        return out.getvalue()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/process", response_model=ProcessResponse)
async def process(
    image: Annotated[UploadFile, File()],
    x_internal_key: Annotated[str | None, Header()] = None,
) -> ProcessResponse:
    _verify_internal_key(x_internal_key)
    _validate_content_type(image.content_type)

    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="empty image",
        )
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"image exceeds max size of {MAX_IMAGE_BYTES} bytes",
        )

    try:
        orientation = detect_orientation(image_bytes)
    except Exception:
        logger.exception("orient failed")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="orientation upstream failure",
        ) from None

    rotated_bytes = _rotate_image_bytes(image_bytes, orientation.rotation_degrees)

    try:
        classification = classify_card(rotated_bytes)
    except ClassifyError:
        logger.exception("classify failed to parse after retry")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="classify response unparseable",
        ) from None
    except Exception:
        logger.exception("classify failed")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="classify upstream failure",
        ) from None

    return ProcessResponse(
        player=classification.player,
        team=classification.team,
        card_number=classification.card_number,
        side=classification.side,
        rotation_degrees=orientation.rotation_degrees,
        orient_confidence=orientation.confidence,
        text_count=orientation.text_count,
    )
