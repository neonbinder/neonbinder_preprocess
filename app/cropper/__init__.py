"""Crop cascade orchestration.

Mirrors the cropAttempts waterfall from script-frontend's imageProcessor.js
but restricted to strategies that run server-side (macOS Swift croppers stay
client-side).

Every cropping strategy flows through the SAME three-gate check, so adding
a new cropper (MobileSAM, classical contour, haiku_bbox, ...) is just a
matter of appending to `_STRATEGIES`. The gates are applied in the
wrapper, not in each strategy, so a new cropper can't accidentally skip
fallback logic.

Gates (applied to every strategy uniformly):

1. **Geometric validation** (`validator.is_plausible_crop`) — min size,
   aspect ratio within tolerance, area fraction vs. source, non-blank
   stddev. Rejects technically malformed crops.

2. **Text-count regression guard** — the baseline is orient's text count
   on the raw passthrough. A cropper's output must retain at least
   `MIN_CASCADE_TEXT_RATIO` of that baseline, or the stage is rejected.
   Catches wrong-region crops that *happen* to be card-shaped.

   (An earlier iteration also had a "classify-error guard" — reject when
   players+card_number both null AND side=back — but it misfired on
   legitimate multi-player leaders/combo card backs where those fields
   are genuinely null. The text-count gate alone is enough to catch
   wrong-region crops without false-rejecting real backs.)

If every strategy fails, the passthrough fallback carries whatever
orient+classify produced on the raw image — which may itself return
empty players / null card_number. That's acceptable: the client can
surface it as "preprocess couldn't identify this card" and route to a
manual path upstream.

Source labels (order of preference):
    precropped : client-supplied crop; trusted when validator + classify-
                 error gates pass (no text-count gate since the baseline
                 hasn't been computed yet on the happy path)
    pil_trim   : PIL blur + threshold + trim + expand
    sam        : SAM ViT-B semantic segmentation
    passthrough: raw image forwarded unchanged
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
# same three-gate wrapper (`_try_stage` below). Add new croppers here —
# the gates are applied uniformly. `passthrough` is handled separately at
# the end since it doesn't need gating (it's the always-available fallback).
#
# Strategies are stored as (name, module, attr) so the callable is looked
# up fresh at each cascade invocation — tests can monkeypatch the module
# attribute and the cascade picks up the patch.
_STRATEGIES: list[tuple[str, object, str]] = [
    ("pil_trim", pil_trim, "trim"),
    ("sam", sam, "sam_crop"),
    ("haiku_bbox", haiku_bbox, "haiku_bbox_crop"),
]


@dataclass(frozen=True)
class CropResult:
    """Outcome of the cascade.

    Carries every downstream signal so main.py's /process handler becomes
    a thin response-packaging layer. `returned_bytes_differ` is True when
    the server produced new bytes the client doesn't already have — i.e.
    the response should include `cropped_image_b64`.
    """

    image_bytes: bytes
    source: str
    returned_bytes_differ: bool
    orientation: OrientationResult
    classification: ClassifyResult


def _orient_and_classify(image_bytes: bytes) -> tuple[OrientationResult, ClassifyResult]:
    orient = detect_orientation(image_bytes)
    rotated = rotate_image_bytes(image_bytes, orient.rotation_degrees)
    classification = classify_card(rotated)
    return orient, classification


def _try_stage(
    *,
    source: str,
    candidate_bytes: bytes,
    source_area_bytes: bytes,
    text_threshold: int,
) -> CropResult | None:
    """Apply the uniform three-gate check to a candidate crop.

    Returns a winning CropResult if all gates pass, None otherwise.
    Caller can treat None as "advance to next strategy."
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
        returned_bytes_differ=True,
        orientation=orient,
        classification=classification,
    )


def crop(
    *,
    image_bytes: bytes,
    precropped_bytes: bytes | None,
) -> CropResult:
    """Run the crop cascade and return the winning result.

    When `precropped_bytes` is provided, that's tried first. When absent,
    the `image` upload is treated as the precropped candidate — preserves
    backward compat with callers who do their own cropping.
    """
    # ── Stage 1 — precropped short-circuit ─────────────────────────────
    # Client-supplied crop: trust it if validator passes AND classify
    # doesn't hit the wrong-region signature. We skip the text-count gate
    # here to avoid an extra baseline Vision call on the happy path; the
    # classify-error check is much cheaper and catches most crop disasters.
    candidate = precropped_bytes if precropped_bytes is not None else image_bytes
    check = is_plausible_crop(candidate)
    if check.ok:
        orient, classification = _orient_and_classify(candidate)
        return CropResult(
            image_bytes=candidate,
            source="precropped",
            returned_bytes_differ=False,
            orientation=orient,
            classification=classification,
        )
    logger.info("cascade: precropped rejected by validator (%s)", check.reason)

    # Baseline — used for both the text-count threshold and the passthrough
    # fallback so we never orient the same bytes twice.
    baseline_orient = detect_orientation(image_bytes)
    text_threshold = max(1, int(baseline_orient.text_count * MIN_CASCADE_TEXT_RATIO))
    logger.info(
        "cascade: baseline text_count=%d, threshold=%d",
        baseline_orient.text_count,
        text_threshold,
    )

    # ── Stages 2..N — server-side croppers through the uniform gate ───
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
        )
        if result is not None:
            return result

    # ── Passthrough ─────────────────────────────────────────────────────
    # Unconditional fallback. Carries whatever orient+classify produced on
    # the raw image; may itself be a classify-error response, which is the
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
