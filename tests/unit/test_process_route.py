"""Unit tests for POST /process.

Covers: auth (missing / wrong / correct), request validation (content-type,
empty body, oversized body), upstream failure translation (Vision / Anthropic
exceptions → 502), the happy path end-to-end, and the rotation-before-classify
wiring (rotated bytes are what classify sees).
"""

from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import main
from app.classify import ClassifyError, ClassifyResult
from app.main import MAX_IMAGE_BYTES, app
from app.orient import OrientationResult

client = TestClient(app)


@pytest.fixture(autouse=True)
def _set_internal_key(monkeypatch):
    monkeypatch.setenv("INTERNAL_API_KEY", "test-key")


def _jpeg(size: tuple[int, int] = (8, 8), color: str = "white") -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color=color).save(buf, format="JPEG")
    return buf.getvalue()


def _png(size: tuple[int, int] = (8, 8)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color="white").save(buf, format="PNG")
    return buf.getvalue()


def _stub_orient(monkeypatch, rotation=0, confidence=1.0, text_count=5):
    result = OrientationResult(
        rotation_degrees=rotation,
        confidence=confidence,
        text_count=text_count,
    )
    monkeypatch.setattr(main, "detect_orientation", lambda _bytes: result)
    return result


def _stub_classify(
    monkeypatch,
    player="Ichiro",
    team="Mariners",
    card_number="51",
    side="front",
):
    result = ClassifyResult(
        player=player,
        team=team,
        card_number=card_number,
        side=side,
        raw_text="{}",
    )
    calls = []

    def _fake(image_bytes: bytes) -> ClassifyResult:
        calls.append(image_bytes)
        return result

    monkeypatch.setattr(main, "classify_card", _fake)
    return calls


class TestAuth:
    def test_missing_key_returns_401(self):
        response = client.post("/process", files={"image": ("card.jpg", _jpeg(), "image/jpeg")})
        assert response.status_code == 401

    def test_wrong_key_returns_401(self):
        response = client.post(
            "/process",
            headers={"x-internal-key": "wrong"},
            files={"image": ("card.jpg", _jpeg(), "image/jpeg")},
        )
        assert response.status_code == 401

    def test_server_without_key_configured_returns_503(self, monkeypatch):
        monkeypatch.delenv("INTERNAL_API_KEY", raising=False)
        response = client.post(
            "/process",
            headers={"x-internal-key": "test-key"},
            files={"image": ("card.jpg", _jpeg(), "image/jpeg")},
        )
        assert response.status_code == 503


class TestRequestValidation:
    def test_unsupported_content_type_returns_415(self, monkeypatch):
        _stub_orient(monkeypatch)
        _stub_classify(monkeypatch)
        response = client.post(
            "/process",
            headers={"x-internal-key": "test-key"},
            files={"image": ("card.txt", b"not an image", "text/plain")},
        )
        assert response.status_code == 415

    def test_empty_image_returns_400(self, monkeypatch):
        _stub_orient(monkeypatch)
        _stub_classify(monkeypatch)
        response = client.post(
            "/process",
            headers={"x-internal-key": "test-key"},
            files={"image": ("card.jpg", b"", "image/jpeg")},
        )
        assert response.status_code == 400

    def test_oversized_image_returns_413(self, monkeypatch):
        _stub_orient(monkeypatch)
        _stub_classify(monkeypatch)
        # Construct a payload one byte above the max. Content doesn't need to
        # be a valid image — the size check runs before any parsing.
        oversized = b"x" * (MAX_IMAGE_BYTES + 1)
        response = client.post(
            "/process",
            headers={"x-internal-key": "test-key"},
            files={"image": ("card.jpg", oversized, "image/jpeg")},
        )
        assert response.status_code == 413

    def test_missing_file_field_returns_422(self):
        response = client.post(
            "/process",
            headers={"x-internal-key": "test-key"},
        )
        assert response.status_code == 422


class TestUpstreamFailures:
    def test_orient_failure_returns_502(self, monkeypatch):
        def _boom(_bytes):
            raise RuntimeError("vision api down")

        monkeypatch.setattr(main, "detect_orientation", _boom)
        _stub_classify(monkeypatch)

        response = client.post(
            "/process",
            headers={"x-internal-key": "test-key"},
            files={"image": ("card.jpg", _jpeg(), "image/jpeg")},
        )
        assert response.status_code == 502
        assert "orientation" in response.json()["detail"].lower()

    def test_classify_failure_returns_502(self, monkeypatch):
        _stub_orient(monkeypatch)

        def _boom(_bytes):
            raise RuntimeError("anthropic api down")

        monkeypatch.setattr(main, "classify_card", _boom)

        response = client.post(
            "/process",
            headers={"x-internal-key": "test-key"},
            files={"image": ("card.jpg", _jpeg(), "image/jpeg")},
        )
        assert response.status_code == 502
        assert "classify" in response.json()["detail"].lower()

    def test_classify_unparseable_returns_502_with_specific_detail(self, monkeypatch):
        _stub_orient(monkeypatch)

        def _boom(_bytes):
            raise ClassifyError("model babbled")

        monkeypatch.setattr(main, "classify_card", _boom)

        response = client.post(
            "/process",
            headers={"x-internal-key": "test-key"},
            files={"image": ("card.jpg", _jpeg(), "image/jpeg")},
        )
        assert response.status_code == 502
        assert "unparse" in response.json()["detail"].lower()


class TestHappyPath:
    def test_returns_combined_orient_and_classify_fields(self, monkeypatch):
        _stub_orient(monkeypatch, rotation=0, confidence=0.9, text_count=12)
        _stub_classify(monkeypatch, player="Jeter", team="Yankees", card_number="2")

        response = client.post(
            "/process",
            headers={"x-internal-key": "test-key"},
            files={"image": ("card.jpg", _jpeg(), "image/jpeg")},
        )

        assert response.status_code == 200
        body = response.json()
        assert body == {
            "player": "Jeter",
            "team": "Yankees",
            "card_number": "2",
            "side": "front",
            "rotation_degrees": 0,
            "orient_confidence": 0.9,
            "text_count": 12,
            # The 8x8 fixture is too small to pass the validator, so the
            # cascade falls through to passthrough. Bytes returned ==
            # bytes uploaded, so no b64 payload.
            "cropped_source": "passthrough",
            "cropped_image_b64": None,
        }

    def test_accepts_png_content_type(self, monkeypatch):
        _stub_orient(monkeypatch)
        _stub_classify(monkeypatch)

        response = client.post(
            "/process",
            headers={"x-internal-key": "test-key"},
            files={"image": ("card.png", _png(), "image/png")},
        )
        assert response.status_code == 200

    def test_rotation_applied_before_classify(self, monkeypatch):
        _stub_orient(monkeypatch, rotation=90)
        classify_inputs = _stub_classify(monkeypatch)

        original_bytes = _jpeg((40, 20))  # wide image, 40x20
        response = client.post(
            "/process",
            headers={"x-internal-key": "test-key"},
            files={"image": ("card.jpg", original_bytes, "image/jpeg")},
        )

        assert response.status_code == 200
        assert len(classify_inputs) == 1
        # The bytes seen by classify must not be the originals — rotation
        # happened. We don't assert exact pixels (JPEG re-encode differs),
        # but the image dimensions should have flipped.
        with Image.open(io.BytesIO(classify_inputs[0])) as rotated:
            assert rotated.size == (20, 40)

    def test_no_rotation_passthrough(self, monkeypatch):
        _stub_orient(monkeypatch, rotation=0)
        classify_inputs = _stub_classify(monkeypatch)

        original = _jpeg((8, 8))
        client.post(
            "/process",
            headers={"x-internal-key": "test-key"},
            files={"image": ("card.jpg", original, "image/jpeg")},
        )

        assert classify_inputs[0] == original


class TestPrecroppedField:
    """Coverage for the optional `precropped` multipart field added in slice 2a."""

    @staticmethod
    def _card_bytes(size: tuple[int, int] = (500, 700)) -> bytes:
        import random

        rng = random.Random(size[0] + size[1])
        raw = bytes(rng.randint(0, 255) for _ in range(size[0] * size[1] * 3))
        out = io.BytesIO()
        Image.frombytes("RGB", size, raw).save(out, format="JPEG", quality=85)
        return out.getvalue()

    def test_valid_precropped_wins_cascade(self, monkeypatch):
        _stub_orient(monkeypatch)
        classify_inputs = _stub_classify(monkeypatch)

        precropped = self._card_bytes()
        response = client.post(
            "/process",
            headers={"x-internal-key": "test-key"},
            files={
                "image": ("raw.jpg", _jpeg((40, 20)), "image/jpeg"),
                "precropped": ("crop.jpg", precropped, "image/jpeg"),
            },
        )

        assert response.status_code == 200, response.text
        body = response.json()
        assert body["cropped_source"] == "precropped"
        # When source == precropped, client already has the bytes — no b64.
        assert body["cropped_image_b64"] is None
        # Classify should see the precropped bytes (no trimming) before rotation.
        assert len(classify_inputs) == 1

    def test_precropped_wrong_content_type_returns_415(self, monkeypatch):
        _stub_orient(monkeypatch)
        _stub_classify(monkeypatch)

        response = client.post(
            "/process",
            headers={"x-internal-key": "test-key"},
            files={
                "image": ("raw.jpg", _jpeg(), "image/jpeg"),
                "precropped": ("crop.txt", b"not an image", "text/plain"),
            },
        )
        assert response.status_code == 415
        assert "precropped" in response.json()["detail"]

    def test_empty_precropped_returns_400(self, monkeypatch):
        _stub_orient(monkeypatch)
        _stub_classify(monkeypatch)

        response = client.post(
            "/process",
            headers={"x-internal-key": "test-key"},
            files={
                "image": ("raw.jpg", _jpeg(), "image/jpeg"),
                "precropped": ("crop.jpg", b"", "image/jpeg"),
            },
        )
        assert response.status_code == 400
        assert "empty precropped" in response.json()["detail"]

    def test_oversized_precropped_returns_413(self, monkeypatch):
        from app.main import MAX_IMAGE_BYTES

        _stub_orient(monkeypatch)
        _stub_classify(monkeypatch)

        response = client.post(
            "/process",
            headers={"x-internal-key": "test-key"},
            files={
                "image": ("raw.jpg", _jpeg(), "image/jpeg"),
                "precropped": ("crop.jpg", b"x" * (MAX_IMAGE_BYTES + 1), "image/jpeg"),
            },
        )
        assert response.status_code == 413

    def test_invalid_precropped_falls_through_to_cascade(self, monkeypatch):
        """A tiny precropped fails the validator; cascade runs on image bytes."""
        _stub_orient(monkeypatch)
        _stub_classify(monkeypatch)

        response = client.post(
            "/process",
            headers={"x-internal-key": "test-key"},
            files={
                "image": ("raw.jpg", _jpeg((40, 20)), "image/jpeg"),
                # 100x100 passes content-type but too-small to be a card.
                "precropped": ("crop.jpg", _jpeg((100, 100)), "image/jpeg"),
            },
        )
        assert response.status_code == 200
        body = response.json()
        # With such a small image and precropped both failing, cascade should
        # reach passthrough (pil_trim on a 40x20 image returns junk and is
        # rejected by validator too).
        assert body["cropped_source"] == "passthrough"
        assert body["cropped_image_b64"] is None

    def test_cascade_produced_bytes_are_returned_as_b64(self, monkeypatch):
        """When source != precropped and cascade produced new bytes, they're b64-encoded."""
        _stub_orient(monkeypatch)
        _stub_classify(monkeypatch)

        # Force pil_trim to return known good card-shaped bytes. The cascade
        # will pick it and returned_bytes_differ will be True.
        good_crop = self._card_bytes()
        monkeypatch.setattr("app.cropper.pil_trim.trim", lambda _b: good_crop)

        response = client.post(
            "/process",
            headers={"x-internal-key": "test-key"},
            files={
                "image": ("raw.jpg", _jpeg((40, 20)), "image/jpeg"),
                "precropped": ("crop.jpg", _jpeg((100, 100)), "image/jpeg"),
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["cropped_source"] == "pil_trim"
        assert body["cropped_image_b64"] is not None
        import base64

        assert base64.b64decode(body["cropped_image_b64"]) == good_crop
