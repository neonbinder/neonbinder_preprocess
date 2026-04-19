"""FastAPI entrypoint for the neonbinder-preprocess service.

Slice 1 (shipped): `/health` + `/process`. `/process` accepts a card image
and returns the structured orient + classify result plus the CCW rotation
applied before classification.

Slice 2a (this change): `/process` now optionally accepts a `precropped`
multipart field. When present, the server validates it; if it looks like
a plausible card, the server uses it as-is. Otherwise (or when omitted),
the server runs a crop cascade (currently: PIL trim → passthrough) on
the raw `image` field. The response advertises which source won via
`cropped_source` and, when the server produced new bytes, includes them
as base64 in `cropped_image_b64` so the client can persist the oriented
copy without re-running anything.

Future slices will add `sam` (2b) and `haiku_bbox` (2c) between pil_trim
and passthrough in the cascade.
"""

from __future__ import annotations

import base64
import hmac
import logging
import os
from io import BytesIO
from typing import Annotated

from fastapi import FastAPI, File, Header, HTTPException, UploadFile, status
from PIL import Image
from pydantic import BaseModel

from app import cropper
from app.classify import ClassifyError, classify_card
from app.orient import detect_orientation

logger = logging.getLogger(__name__)

app = FastAPI(title="neonbinder-preprocess", version="0.2.0")

INTERNAL_API_KEY_ENV = "INTERNAL_API_KEY"
MAX_IMAGE_BYTES = 32 * 1024 * 1024
ALLOWED_CONTENT_TYPES = frozenset({"image/jpeg", "image/png", "image/webp"})


class ProcessResponse(BaseModel):
    """Response body for POST /process.

    `rotation_degrees` is the CCW rotation that was applied to the chosen
    crop before classification; clients that store the corrected image
    should apply the same rotation to keep their copy aligned with what
    the model actually saw.

    `cropped_source` tells the client which stage of the crop cascade
    produced the working image. When it is `"precropped"` the client's
    upload was used as-is and `cropped_image_b64` will be null (the client
    already has the bytes on disk). For every other source, the server
    produced new bytes and returns them base64-encoded in
    `cropped_image_b64` so the client can persist them.
    """

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


async def _read_upload(
    upload: UploadFile,
    *,
    field: str,
) -> bytes:
    """Read an UploadFile, enforcing content-type and size gates."""
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
    precropped: Annotated[UploadFile | None, File()] = None,
    x_internal_key: Annotated[str | None, Header()] = None,
) -> ProcessResponse:
    _verify_internal_key(x_internal_key)

    image_bytes = await _read_upload(image, field="image")
    precropped_bytes: bytes | None = None
    if precropped is not None:
        precropped_bytes = await _read_upload(precropped, field="precropped")

    crop_result = cropper.crop(
        image_bytes=image_bytes,
        precropped_bytes=precropped_bytes,
    )

    try:
        orientation = detect_orientation(crop_result.image_bytes)
    except Exception:
        logger.exception("orient failed")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="orientation upstream failure",
        ) from None

    rotated_bytes = _rotate_image_bytes(crop_result.image_bytes, orientation.rotation_degrees)

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

    cropped_image_b64: str | None = None
    if crop_result.returned_bytes_differ:
        cropped_image_b64 = base64.b64encode(crop_result.image_bytes).decode("ascii")

    return ProcessResponse(
        player=classification.player,
        team=classification.team,
        card_number=classification.card_number,
        side=classification.side,
        rotation_degrees=orientation.rotation_degrees,
        orient_confidence=orientation.confidence,
        text_count=orientation.text_count,
        cropped_source=crop_result.source,
        cropped_image_b64=cropped_image_b64,
    )
