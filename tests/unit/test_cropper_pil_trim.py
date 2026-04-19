"""Unit tests for app.cropper.pil_trim.

Focuses on the observable behavior: given a card-on-dark-background input,
the trim extracts the card region and returns JPEG bytes with reasonable
dimensions. Precise pixel-exact matching against a reference image is
brittle, so we assert shape properties instead.
"""

from __future__ import annotations

import io

from PIL import Image, ImageDraw

from app.cropper.pil_trim import BORDER_PX, MAX_LONGEST_EDGE_PX, trim


def _card_on_dark_background(
    *,
    canvas_size: tuple[int, int] = (1200, 1600),
    card_box: tuple[int, int, int, int] = (200, 300, 1000, 1400),
    card_color: str = "white",
) -> bytes:
    """Render a light card on a black canvas.

    Returns JPEG bytes the trim can meaningfully cut.
    """
    img = Image.new("RGB", canvas_size, color="black")
    draw = ImageDraw.Draw(img)
    draw.rectangle(card_box, fill=card_color)
    out = io.BytesIO()
    img.save(out, format="JPEG", quality=90)
    return out.getvalue()


class TestTrim:
    def test_returns_bytes_for_card_on_background(self):
        result = trim(_card_on_dark_background())
        assert result is not None
        # Round-trip through PIL to confirm it decoded as an image.
        with Image.open(io.BytesIO(result)) as out:
            assert out.size[0] > 0 and out.size[1] > 0

    def test_output_includes_border(self):
        # Card is 800 wide × 1100 tall; output should be roughly that plus
        # 2*BORDER_PX on each axis.
        result = trim(_card_on_dark_background())
        assert result is not None
        with Image.open(io.BytesIO(result)) as out:
            w, h = out.size
            assert 800 <= w <= 800 + 2 * BORDER_PX + 10
            assert 1100 <= h <= 1100 + 2 * BORDER_PX + 10

    def test_downscales_very_large_inputs(self):
        # Construct an image well over the downscale threshold; verify the
        # output doesn't exceed the cap on either dimension.
        big = _card_on_dark_background(
            canvas_size=(5000, 6000),
            card_box=(800, 1000, 4200, 5000),
        )
        result = trim(big)
        assert result is not None
        with Image.open(io.BytesIO(result)) as out:
            assert max(out.size) <= MAX_LONGEST_EDGE_PX

    def test_returns_none_for_all_black_image(self):
        img = Image.new("RGB", (1000, 1400), color="black")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        # Nothing above the threshold → getbbox returns None.
        assert trim(buf.getvalue()) is None

    def test_returns_none_for_unreadable_bytes(self):
        assert trim(b"definitely not an image") is None
