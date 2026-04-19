"""Crop cascade orchestration.

Mirrors the cropAttempts waterfall from the script-frontend's imageProcessor.js
but restricted to strategies that run server-side (macOS Swift croppers stay
client-side).

The cascade runs each strategy in order; the first one whose output passes
`validator.is_plausible_crop` wins. Strategies that raise, return None, or
produce a crop that fails validation are skipped silently and the next
strategy runs.

Each strategy is a pure function `(image_bytes) -> bytes | None`. The
cascade itself returns a `CropResult` with the winning source label plus
the resulting bytes.

Source labels (order of preference):
    precropped : client supplied a valid crop candidate
    pil_trim   : PIL blur + threshold + trim + expand (sharp.trim port)
    sam        : [slice 2b] SAM semantic segmentation
    haiku_bbox : [slice 2c] Anthropic Haiku bbox
    passthrough: last resort, input returned unchanged

"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from app.cropper import pil_trim
from app.cropper.validator import is_plausible_crop

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CropResult:
    """Outcome of the cascade.

    image_bytes: final cropped image (may equal the input if source == "passthrough").
    source:      label of the winning strategy.
    returned_bytes_differ: True when image_bytes != the caller's original upload,
                           i.e. the response should include cropped_image_b64.
                           False when source == "precropped" (client already has
                           these exact bytes on disk).
    """

    image_bytes: bytes
    source: str
    returned_bytes_differ: bool


def crop(
    *,
    image_bytes: bytes,
    precropped_bytes: bytes | None,
) -> CropResult:
    """Run the crop cascade and return the first successful result.

    When `precropped_bytes` is provided, that's tried first. When it's absent,
    the `image` upload is treated as the precropped candidate — preserves
    backward compat with slice-1 callers who already do their own cropping
    and send only `image`.
    """
    # Stage 1 — validate the precropped candidate (or image as fallback).
    candidate = precropped_bytes if precropped_bytes is not None else image_bytes
    source_label = "precropped"
    check = is_plausible_crop(candidate)
    if check.ok:
        return CropResult(
            image_bytes=candidate,
            source=source_label,
            returned_bytes_differ=False,
        )
    logger.info("cascade: precropped candidate rejected (%s)", check.reason)

    # Stage 2 — PIL trim on the original image.
    try:
        trimmed = pil_trim.trim(image_bytes)
    except Exception as exc:  # noqa: BLE001
        logger.warning("cascade: pil_trim raised %s", exc)
        trimmed = None
    if trimmed is not None:
        check = is_plausible_crop(trimmed, source_area_bytes=image_bytes)
        if check.ok:
            return CropResult(
                image_bytes=trimmed,
                source="pil_trim",
                returned_bytes_differ=True,
            )
        logger.info("cascade: pil_trim rejected (%s)", check.reason)

    # TODO(slice-2b): insert `sam` strategy here.
    # TODO(slice-2c): insert `haiku_bbox` strategy here.

    # Stage N — passthrough. Return original unchanged so orient+classify have
    # something to work with even when every cropper failed.
    logger.info("cascade: falling through to passthrough")
    return CropResult(
        image_bytes=image_bytes,
        source="passthrough",
        returned_bytes_differ=False,
    )
