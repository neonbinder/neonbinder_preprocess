"""Unit tests for app.cropper.crop — the cascade orchestrator.

Exercises each branch of the waterfall: precropped-passes, precropped-fails
(falls through to pil_trim), no-precropped (image used as candidate),
pil_trim produces result, pil_trim fails → passthrough.
"""

from __future__ import annotations

import io

from PIL import Image

from app.cropper import CropResult, crop


def _card_jpeg(*, size: tuple[int, int] = (500, 700)) -> bytes:
    """A plausible card: card-sized, card-shaped, noisy (not blank)."""
    import random

    rng = random.Random(size[0] * 31 + size[1])
    raw = bytes(rng.randint(0, 255) for _ in range(size[0] * size[1] * 3))
    img = Image.frombytes("RGB", size, raw)
    out = io.BytesIO()
    img.save(out, format="JPEG", quality=85)
    return out.getvalue()


def _tiny_jpeg() -> bytes:
    """Too small to pass the validator — forces cascade fallthrough."""
    return _card_jpeg(size=(100, 140))


class TestCascade:
    def test_precropped_passes_validation_is_used(self):
        image_bytes = _card_jpeg(size=(1200, 1600))  # raw
        precropped = _card_jpeg(size=(500, 700))  # a good crop

        result = crop(image_bytes=image_bytes, precropped_bytes=precropped)

        assert result.source == "precropped"
        assert result.image_bytes == precropped
        assert result.returned_bytes_differ is False

    def test_missing_precropped_uses_image_when_image_validates(self):
        # Slice-1 compat: no precropped, and the image itself looks like a
        # card (already pre-cropped client-side). Cascade should recognize
        # it as the precropped candidate and use it without trimming.
        image_bytes = _card_jpeg(size=(500, 700))

        result = crop(image_bytes=image_bytes, precropped_bytes=None)

        assert result.source == "precropped"
        assert result.image_bytes == image_bytes
        assert result.returned_bytes_differ is False

    def test_invalid_precropped_falls_through_to_pil_trim(self, monkeypatch):
        # Force pil_trim to return a known good crop so we can detect that
        # stage ran instead of skipping straight to passthrough.
        synthetic_trimmed = _card_jpeg(size=(500, 700))
        monkeypatch.setattr("app.cropper.pil_trim.trim", lambda _b: synthetic_trimmed)

        # The "precropped" is too small → validator rejects, cascade advances.
        image_bytes = _card_jpeg(size=(1200, 1600))
        bad_precropped = _tiny_jpeg()

        result = crop(image_bytes=image_bytes, precropped_bytes=bad_precropped)

        assert result.source == "pil_trim"
        assert result.image_bytes == synthetic_trimmed
        assert result.returned_bytes_differ is True

    def test_pil_trim_output_rejected_falls_through_to_passthrough(self, monkeypatch):
        # pil_trim returns something, but it fails validation (too small).
        # Cascade should fall through to passthrough.
        monkeypatch.setattr("app.cropper.pil_trim.trim", lambda _b: _tiny_jpeg())

        image_bytes = _card_jpeg(size=(1200, 1600))
        bad_precropped = _tiny_jpeg()

        result = crop(image_bytes=image_bytes, precropped_bytes=bad_precropped)

        assert result.source == "passthrough"
        assert result.image_bytes == image_bytes
        assert result.returned_bytes_differ is False

    def test_pil_trim_returns_none_falls_through_to_passthrough(self, monkeypatch):
        monkeypatch.setattr("app.cropper.pil_trim.trim", lambda _b: None)

        image_bytes = _card_jpeg(size=(1200, 1600))
        bad_precropped = _tiny_jpeg()

        result = crop(image_bytes=image_bytes, precropped_bytes=bad_precropped)

        assert result.source == "passthrough"
        assert result.image_bytes == image_bytes

    def test_pil_trim_raises_falls_through_to_passthrough(self, monkeypatch):
        def _boom(_b):
            raise RuntimeError("pil_trim exploded")

        monkeypatch.setattr("app.cropper.pil_trim.trim", _boom)

        image_bytes = _card_jpeg(size=(1200, 1600))
        bad_precropped = _tiny_jpeg()

        result = crop(image_bytes=image_bytes, precropped_bytes=bad_precropped)

        # Exception in a strategy must not kill the cascade — fall through.
        assert result.source == "passthrough"
        assert result.image_bytes == image_bytes

    def test_crop_result_is_immutable_dataclass(self):
        r = CropResult(image_bytes=b"x", source="precropped", returned_bytes_differ=False)
        try:
            r.source = "other"  # type: ignore[misc]
        except Exception:
            return
        raise AssertionError("CropResult should be frozen")
