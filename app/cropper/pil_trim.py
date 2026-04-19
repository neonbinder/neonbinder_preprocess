"""PIL-based trim — port of script-frontend's sharp.trim strategy.

Works well on photos of cards on a near-solid (usually black or dark)
background: blur to smooth noise, threshold out dark pixels, take the
bounding box of the remaining content, then add a small border back so the
next stages (orient, classify) don't clip the card edge.

This replaces both `sharp.trim` and `magick.fuzz.trim` from the original
cascade — they solve the same problem (background bleed around the card)
at different thresholds, and PIL + ImageChops is close enough to either
that running both is redundant.
"""

from __future__ import annotations

from io import BytesIO

from PIL import Image, ImageFilter

# Downscale very large images before processing — libvips segfaults on 50MP+
# inputs (per the script-frontend comment) and PIL is slow too. 3000px longest
# edge is enough resolution to find card edges reliably.
MAX_LONGEST_EDGE_PX = 3000

# Gaussian blur radius before thresholding. Smooths JPEG artifacts + film
# grain without dissolving real card edges.
BLUR_RADIUS = 0.8

# Threshold for "is this pixel background?" — values at or below this on the
# grayscale channel are treated as background. 180 matches sharp.trim's
# threshold (sharp uses a 0..255 scale with 180 meaning "trim anything
# darker than this").
DARK_THRESHOLD = 180

# Border added back around the bounding box so the orient + classify stages
# have breathing room. In pixels.
BORDER_PX = 10


def _maybe_downscale(img: Image.Image) -> Image.Image:
    longest = max(img.size)
    if longest <= MAX_LONGEST_EDGE_PX:
        return img
    scale = MAX_LONGEST_EDGE_PX / longest
    new_size = (int(img.width * scale), int(img.height * scale))
    return img.resize(new_size, Image.Resampling.LANCZOS)


def trim(image_bytes: bytes) -> bytes | None:
    """Run PIL-based background trim on the image.

    Returns the trimmed JPEG bytes, or None if no meaningful bounding box
    could be found (e.g. the whole image is below the dark threshold, or
    the input is unreadable).
    """
    try:
        img = Image.open(BytesIO(image_bytes))
        img.load()
    except Exception:  # noqa: BLE001
        return None

    img = img.convert("RGB")
    img = _maybe_downscale(img)

    blurred = img.filter(ImageFilter.GaussianBlur(radius=BLUR_RADIUS))
    gray = blurred.convert("L")

    # Create a "foreground" mask: pixels brighter than the dark threshold are
    # content; everything else is background. getbbox() returns the bounding
    # box of non-black pixels, so we threshold into a binary image first.
    mask = gray.point(lambda p: 255 if p > DARK_THRESHOLD else 0, mode="L")
    bbox = mask.getbbox()
    if bbox is None:
        return None

    left, top, right, bottom = bbox
    if right - left < 1 or bottom - top < 1:
        return None

    cropped = img.crop(bbox)

    # Add a black border so the cropper doesn't shave off a pixel-thin slice
    # of the card edge. `ImageChops.offset` + expand would also work; this is
    # clearer.
    w, h = cropped.size
    bordered = Image.new("RGB", (w + 2 * BORDER_PX, h + 2 * BORDER_PX), color="black")
    bordered.paste(cropped, (BORDER_PX, BORDER_PX))

    out = BytesIO()
    bordered.save(out, format="JPEG", quality=90)
    return out.getvalue()
