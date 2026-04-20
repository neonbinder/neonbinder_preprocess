"""Unit tests for app.cropper.crop — the cascade orchestrator.

Stubs the four building blocks the cascade talks to:
    pil_trim.trim, sam.sam_crop      — crop strategies
    detect_orientation, classify_card — quality gate inputs

Each test exercises one branch of the three-gate wrapper:
    1. geometric validator
    2. text-count regression guard (vs. passthrough baseline)
    3. classify-error guard (player+card_number null + side=back)
"""

from __future__ import annotations

import io

import pytest
from PIL import Image

from app import cropper
from app.classify import ClassifyResult
from app.cropper import CropResult, crop
from app.orient import OrientationResult


def _card_jpeg(*, size: tuple[int, int] = (500, 700)) -> bytes:
    import random

    rng = random.Random(size[0] * 31 + size[1])
    raw = bytes(rng.randint(0, 255) for _ in range(size[0] * size[1] * 3))
    img = Image.frombytes("RGB", size, raw)
    out = io.BytesIO()
    img.save(out, format="JPEG", quality=85)
    return out.getvalue()


def _tiny_jpeg() -> bytes:
    return _card_jpeg(size=(100, 140))


def _orient(
    *,
    text_count: int = 10,
    rotation: int = 0,
    confidence: float = 1.0,
) -> OrientationResult:
    return OrientationResult(
        rotation_degrees=rotation, confidence=confidence, text_count=text_count
    )


def _classify(
    *,
    player: str | None = "Ichiro",
    team: str | None = "Mariners",
    card_number: str | None = "51",
    side: str = "front",
) -> ClassifyResult:
    return ClassifyResult(
        player=player, team=team, card_number=card_number, side=side, raw_text="{}"
    )


def _classify_error() -> ClassifyResult:
    """The null/null/back pattern the cascade treats as a crop error."""
    return _classify(player=None, card_number=None, side="back")


@pytest.fixture
def stub_orient(monkeypatch):
    """Factory: install an orient stub that returns the same result for every call."""

    def _install(result: OrientationResult | None = None) -> list[bytes]:
        result = result or _orient()
        calls: list[bytes] = []

        def _fake(b: bytes) -> OrientationResult:
            calls.append(b)
            return result

        monkeypatch.setattr(cropper, "detect_orientation", _fake)
        return calls

    return _install


@pytest.fixture
def stub_orient_by_call(monkeypatch):
    """Install an orient stub that returns successive queued results in order."""

    def _install(*results: OrientationResult) -> list[bytes]:
        queue = list(results)
        calls: list[bytes] = []

        def _fake(b: bytes) -> OrientationResult:
            calls.append(b)
            return queue.pop(0) if queue else _orient()

        monkeypatch.setattr(cropper, "detect_orientation", _fake)
        return calls

    return _install


@pytest.fixture
def stub_classify(monkeypatch):
    """Install a classify stub that returns successive queued results in order."""

    def _install(*results: ClassifyResult) -> list[bytes]:
        queue = list(results)
        calls: list[bytes] = []

        def _fake(b: bytes) -> ClassifyResult:
            calls.append(b)
            return queue.pop(0) if queue else _classify()

        monkeypatch.setattr(cropper, "classify_card", _fake)
        return calls

    return _install


class TestPrecroppedShortCircuit:
    def test_valid_precropped_with_good_classify_is_used(self, stub_orient, stub_classify):
        stub_orient()
        stub_classify(_classify())

        image = _card_jpeg(size=(1200, 1600))
        precropped = _card_jpeg(size=(500, 700))

        result = crop(image_bytes=image, precropped_bytes=precropped)

        assert result.source == "precropped"
        assert result.image_bytes == precropped
        assert result.returned_bytes_differ is False
        assert result.classification.player == "Ichiro"

    def test_missing_precropped_uses_image_when_image_validates(self, stub_orient, stub_classify):
        # Slice-1 compat: no precropped, image looks card-shaped → short-circuit.
        stub_orient()
        stub_classify(_classify())

        image = _card_jpeg(size=(500, 700))

        result = crop(image_bytes=image, precropped_bytes=None)

        assert result.source == "precropped"
        assert result.image_bytes == image

    def test_precropped_classify_error_falls_through_to_pil_trim(
        self, monkeypatch, stub_orient, stub_classify
    ):
        # Precropped passes the validator but classify says null/null/back.
        # Cascade should advance to pil_trim, which returns a good crop.
        good_trim = _card_jpeg(size=(500, 700))
        monkeypatch.setattr("app.cropper.pil_trim.trim", lambda _b: good_trim)

        stub_orient()  # all calls return text_count=10
        stub_classify(_classify_error(), _classify())
        # 1st classify: precropped → error → fall through
        # 2nd classify: pil_trim → good → win

        image = _card_jpeg(size=(1200, 1600))
        precropped = _card_jpeg(size=(500, 700))

        result = crop(image_bytes=image, precropped_bytes=precropped)

        assert result.source == "pil_trim"
        assert result.image_bytes == good_trim
        assert result.returned_bytes_differ is True


class TestPilTrimStage:
    def test_valid_pil_trim_with_sufficient_text_wins(
        self, monkeypatch, stub_orient, stub_classify
    ):
        good = _card_jpeg(size=(500, 700))
        monkeypatch.setattr("app.cropper.pil_trim.trim", lambda _b: good)

        # Every orient call returns text_count=10. Threshold = 0.8 * 10 = 8,
        # so pil_trim's 10 clears the gate.
        stub_orient()
        stub_classify(_classify())

        image = _card_jpeg(size=(1200, 1600))
        bad_precropped = _tiny_jpeg()

        result = crop(image_bytes=image, precropped_bytes=bad_precropped)

        assert result.source == "pil_trim"

    def test_pil_trim_text_count_drop_falls_through(
        self, monkeypatch, stub_orient_by_call, stub_classify
    ):
        """pil_trim passes validator but drops too much text → fall through."""
        good_sam = _card_jpeg(size=(500, 700))
        monkeypatch.setattr("app.cropper.pil_trim.trim", lambda _b: good_sam)
        monkeypatch.setattr("app.cropper.sam.sam_crop", lambda _b: good_sam)

        # Baseline text=10 → threshold=8. pil_trim's text=5 fails the gate.
        # Then cascade tries SAM (stubbed with text=10) → wins.
        stub_orient_by_call(
            _orient(text_count=10),  # baseline (passthrough)
            _orient(text_count=5),  # pil_trim output
            _orient(text_count=10),  # sam output
        )
        stub_classify(_classify())  # sam's classify (pil_trim never reached classify)

        image = _card_jpeg(size=(1200, 1600))
        bad_precropped = _tiny_jpeg()

        result = crop(image_bytes=image, precropped_bytes=bad_precropped)

        assert result.source == "sam"

    def test_pil_trim_classify_error_falls_through(self, monkeypatch, stub_orient, stub_classify):
        """pil_trim passes validator + text-count but classify is a wrong-region signal."""
        good = _card_jpeg(size=(500, 700))
        monkeypatch.setattr("app.cropper.pil_trim.trim", lambda _b: good)
        monkeypatch.setattr("app.cropper.sam.sam_crop", lambda _b: good)

        stub_orient()  # all calls return text_count=10
        stub_classify(_classify_error(), _classify())
        # 1st classify: pil_trim → null/null/back → fall through
        # 2nd classify: sam → good → win

        image = _card_jpeg(size=(1200, 1600))
        bad_precropped = _tiny_jpeg()

        result = crop(image_bytes=image, precropped_bytes=bad_precropped)

        assert result.source == "sam"


class TestSamStage:
    def test_valid_sam_wins_when_pil_trim_empty(self, monkeypatch, stub_orient, stub_classify):
        good = _card_jpeg(size=(500, 700))
        monkeypatch.setattr("app.cropper.pil_trim.trim", lambda _b: None)
        monkeypatch.setattr("app.cropper.sam.sam_crop", lambda _b: good)

        stub_orient()
        stub_classify(_classify())

        image = _card_jpeg(size=(1200, 1600))
        bad_precropped = _tiny_jpeg()

        result = crop(image_bytes=image, precropped_bytes=bad_precropped)

        assert result.source == "sam"

    def test_sam_classify_error_falls_through_to_passthrough(
        self, monkeypatch, stub_orient, stub_classify
    ):
        good = _card_jpeg(size=(500, 700))
        monkeypatch.setattr("app.cropper.pil_trim.trim", lambda _b: None)
        monkeypatch.setattr("app.cropper.sam.sam_crop", lambda _b: good)

        stub_orient()
        # Two classify calls:
        #   1. sam → classify_error → fall through
        #   2. passthrough → good → packaged (baseline path)
        stub_classify(_classify_error(), _classify())

        image = _card_jpeg(size=(1200, 1600))
        bad_precropped = _tiny_jpeg()

        result = crop(image_bytes=image, precropped_bytes=bad_precropped)

        assert result.source == "passthrough"
        assert result.image_bytes == image

    def test_sam_raises_falls_through_to_passthrough(self, monkeypatch, stub_orient, stub_classify):
        monkeypatch.setattr("app.cropper.pil_trim.trim", lambda _b: None)

        def _boom(_b):
            raise RuntimeError("SAM crashed")

        monkeypatch.setattr("app.cropper.sam.sam_crop", _boom)

        stub_orient()
        stub_classify(_classify())

        image = _card_jpeg(size=(1200, 1600))
        bad_precropped = _tiny_jpeg()

        result = crop(image_bytes=image, precropped_bytes=bad_precropped)

        assert result.source == "passthrough"


class TestPassthroughFallback:
    def test_all_stages_fail_returns_passthrough_with_its_own_classify(
        self, monkeypatch, stub_orient, stub_classify
    ):
        """Every stage rejects → passthrough's classify rides through (even if an error)."""
        monkeypatch.setattr("app.cropper.pil_trim.trim", lambda _b: None)
        monkeypatch.setattr("app.cropper.sam.sam_crop", lambda _b: None)

        stub_orient()
        stub_classify(_classify_error())
        # Only one classify call: on the passthrough's rotated bytes.

        image = _card_jpeg(size=(1200, 1600))
        bad_precropped = _tiny_jpeg()

        result = crop(image_bytes=image, precropped_bytes=bad_precropped)

        assert result.source == "passthrough"
        assert result.classification.player is None
        assert result.classification.card_number is None
        assert result.classification.side == "back"

    def test_sam_output_rejected_by_validator_passthrough_wins(
        self, monkeypatch, stub_orient, stub_classify
    ):
        """SAM returns bytes that fail the validator (too small) → fall through."""
        monkeypatch.setattr("app.cropper.pil_trim.trim", lambda _b: None)
        monkeypatch.setattr("app.cropper.sam.sam_crop", lambda _b: _tiny_jpeg())

        stub_orient()
        stub_classify(_classify())

        image = _card_jpeg(size=(1200, 1600))
        bad_precropped = _tiny_jpeg()

        result = crop(image_bytes=image, precropped_bytes=bad_precropped)

        assert result.source == "passthrough"


class TestCropResultShape:
    def test_is_immutable_dataclass(self, stub_orient, stub_classify):
        stub_orient()
        stub_classify(_classify())
        result = crop(image_bytes=_card_jpeg(), precropped_bytes=None)
        # Frozen dataclass raises FrozenInstanceError (subclass of AttributeError)
        # on any field mutation.
        with pytest.raises(AttributeError):
            result.source = "other"  # type: ignore[misc]

    def test_carries_orientation_and_classification(self, stub_orient, stub_classify):
        stub_orient(_orient(text_count=42, rotation=90, confidence=0.77))
        stub_classify(_classify(player="Jeter", team="Yankees", card_number="2"))
        result: CropResult = crop(image_bytes=_card_jpeg(), precropped_bytes=None)
        assert result.orientation.text_count == 42
        assert result.orientation.rotation_degrees == 90
        assert result.classification.player == "Jeter"
        assert result.classification.card_number == "2"
