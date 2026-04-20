"""Crop cascade orchestration.

Mirrors the cropAttempts waterfall from script-frontend's imageProcessor.js
but restricted to strategies that run server-side (macOS Swift croppers stay
client-side).

Every cropping strategy — including the client-supplied `precropped` —
flows through the SAME two-gate check, so adding a new cropper (MobileSAM,
classical contour, ...) is just a matter of appending to `_STRATEGIES`.
The gates are applied in the wrapper, not in each strategy, so a new
cropper can't accidentally skip fallback logic.

Gates (applied to every strategy uniformly):

1. **Geometric validation** (`validator.is_plausible_crop`) — min size,
   aspect ratio within tolerance, area fraction vs. source, non-blank
   stddev. Rejects technically malformed crops.

2. **Text-count regression guard** — the baseline is orient's text count
   on the raw passthrough. A cropper's output must retain at least
   `MIN_CASCADE_TEXT_RATIO` of that baseline, or the stage is rejected.
   Catches wrong-region crops that *happen* to be card-shaped.

If every strategy fails, the passthrough fallback carries whatever
orient+classify produced on the raw image. The client can surface an
empty-players / null-card_number response as "preprocess couldn't
identify this card" and route to a manual path upstream.

Source labels (order of preference):
    precropped      : client-supplied crop, or the raw upload as fallback
    pil_trim_dark   : PIL blur + threshold + trim (card lighter than bg)
    pil_trim_light  : PIL blur + threshold + trim (card darker than bg)
    sam             : SAM ViT-B semantic segmentation
    haiku_bbox      : Anthropic Haiku bounding-box crop
    passthrough     : raw image forwarded unchanged
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

from app.classify import ClassifyResult, classify_card
from app.cropper import haiku_bbox, pil_trim, sam
from app.cropper._utils import rotate_image_bytes
from app.cropper.validator import is_plausible_crop
from app.orient import OrientationResult, detect_orientation

logger = logging.getLogger(__name__)

# A cascade stage must retain at least this fraction of the baseline orient
# text count or the stage is rejected. 0.8 tolerates Vision's jitter between
# similar crops without letting a wrong-region crop slip through.
MIN_CASCADE_TEXT_RATIO = 0.8

# Type for a crop strategy: takes raw image bytes, returns cropped bytes or None.
CropStrategy = Callable[[bytes], bytes | None]

# Ordered list of server-side crop strategies. Each one flows through the
# same two-gate wrapper (`_try_stage` below). Stored as (name, module, attr)
# so the callable is looked up fresh at each cascade invocation — tests
# monkey-patch the module attribute and the cascade picks up the patch.
#
# `precropped` is NOT in this list — it's handled as stage 1 inside `crop()`
# because the candidate bytes come from a kwarg, not from applying a
# function to `image_bytes`. Gate application is identical.
_STRATEGIES: list[tuple[str, object, str]] = [
    ("pil_trim_dark", pil_trim, "trim_dark"),
    ("pil_trim_light", pil_trim, "trim_light"),
    ("sam", sam, "sam_crop"),
    ("haiku_bbox", haiku_bbox, "haiku_bbox_crop"),
]


@dataclass(frozen=True)
class CropResult:
    """Outcome of the cascade.

    `returned_bytes_differ` is True when the server produced new bytes the
    client doesn't already have — i.e. the response should include
    `cropped_image_b64`. False for precropped (client uploaded those exact
    bytes) and passthrough (client uploaded the raw image).
    """

    image_bytes: bytes
    source: str
    returned_bytes_differ: bool
    orientation: OrientationResult
    classification: ClassifyResult


def _try_stage(
    *,
    source: str,
    candidate_bytes: bytes,
    source_area_bytes: bytes,
    text_threshold: int,
    returned_bytes_differ: bool,
) -> CropResult | None:
    """Apply the uniform two-gate check to a candidate crop.

    Returns a winning CropResult if all gates pass, None otherwise.
    Caller can treat None as "advance to the next strategy."
    """
    check = is_plausible_crop(candidate_bytes, source_area_bytes=source_area_bytes)
    if not check.ok:
        logger.info("cascade: %s rejected by validator (%s)", source, check.reason)
        return None

    orient = detect_orientation(candidate_bytes)
    if orient.text_count < text_threshold:
        logger.info(
            "cascade: %s text_count=%d below threshold=%d, falling through",
            source,
            orient.text_count,
            text_threshold,
        )
        return None

    rotated = rotate_image_bytes(candidate_bytes, orient.rotation_degrees)
    classification = classify_card(rotated)

    return CropResult(
        image_bytes=candidate_bytes,
        source=source,
        returned_bytes_differ=returned_bytes_differ,
        orientation=orient,
        classification=classification,
    )


def crop(
    *,
    image_bytes: bytes,
    precropped_bytes: bytes | None,
) -> CropResult:
    """Run the crop cascade and return the winning result.

    When `precropped_bytes` is provided, that's tried first via `_try_stage`.
    When absent, the `image` upload is used as the stage-1 candidate (slice-1
    backward compat for callers who already cropped client-side).

    The baseline orient on `image_bytes` is computed up front — one extra
    Vision call relative to the old precropped-short-circuit path — so the
    text-count gate applies uniformly to every stage, including precropped.
    """
    # ── Baseline — used for the text-count threshold AND as the passthrough
    # fallback orient. Computed once, reused throughout.
    baseline_orient = detect_orientation(image_bytes)
    text_threshold = max(1, int(baseline_orient.text_count * MIN_CASCADE_TEXT_RATIO))
    logger.info(
        "cascade: baseline text_count=%d, threshold=%d",
        baseline_orient.text_count,
        text_threshold,
    )

    # ── Stage 1 — precropped (or raw image as candidate) through the uniform gate.
    # The client uploaded these exact bytes either way, so returned_bytes_differ
    # stays False regardless of which branch wins.
    stage1_candidate = precropped_bytes if precropped_bytes is not None else image_bytes
    result = _try_stage(
        source="precropped",
        candidate_bytes=stage1_candidate,
        source_area_bytes=image_bytes,
        text_threshold=text_threshold,
        returned_bytes_differ=False,
    )
    if result is not None:
        return result

    # ── Stages 2..N — server-side croppers through the same uniform gate.
    for source, module, fn_name in _STRATEGIES:
        strategy_fn: CropStrategy = getattr(module, fn_name)
        try:
            produced = strategy_fn(image_bytes)
        except Exception as exc:  # noqa: BLE001
            logger.warning("cascade: %s raised %s", source, exc)
            continue
        if produced is None:
            continue

        result = _try_stage(
            source=source,
            candidate_bytes=produced,
            source_area_bytes=image_bytes,
            text_threshold=text_threshold,
            returned_bytes_differ=True,
        )
        if result is not None:
            return result

    # ── Passthrough ─────────────────────────────────────────────────────
    # Unconditional fallback. Carries whatever orient+classify produced on
    # the raw image. May itself be empty-players / null card_number — the
    # honest "preprocess couldn't identify this card" signal.
    logger.info("cascade: falling through to passthrough")
    rotated = rotate_image_bytes(image_bytes, baseline_orient.rotation_degrees)
    passthrough_classification = classify_card(rotated)
    return CropResult(
        image_bytes=image_bytes,
        source="passthrough",
        returned_bytes_differ=False,
        orientation=baseline_orient,
        classification=passthrough_classification,
    )
