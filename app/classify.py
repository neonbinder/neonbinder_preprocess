"""Card classification via Anthropic Claude Haiku.

Sends the card image to Haiku with a strict-JSON prompt, parses the response,
and returns structured fields. Retries once on parse failure with a firmer
reminder to emit raw JSON.
"""

from __future__ import annotations

import base64
import json
import logging
from dataclasses import dataclass
from io import BytesIO

import anthropic
from PIL import Image

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-haiku-4-5-20251001"
MAX_TOKENS = 512

PROMPT = """You are analyzing a trading card photo. Extract key fields as a strict JSON object
with these keys:
- "player": the main player or subject's name (string, or null if not visible)
- "team": the team name if visible on the card, else null
- "card_number": the card number as printed (e.g. "25", "RC-12"), string or null
- "side": either "front" or "back"

Respond with ONLY the JSON. No preamble, no code fences, no trailing text."""

RETRY_PROMPT_SUFFIX = (
    "\n\nYour previous response could not be parsed as JSON. "
    "Return ONLY a raw JSON object with the four keys above — "
    "no markdown, no code fences, no explanation."
)

_MEDIA_TYPE_BY_PIL_FORMAT = {
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "WEBP": "image/webp",
    "GIF": "image/gif",
}


@dataclass(frozen=True)
class ClassifyResult:
    """Extracted card fields plus the model's raw response for debugging."""

    player: str | None
    team: str | None
    card_number: str | None
    side: str
    raw_text: str


class ClassifyError(RuntimeError):
    """Raised when the model's response cannot be parsed after retries."""


def _detect_media_type(image_bytes: bytes) -> str:
    try:
        with Image.open(BytesIO(image_bytes)) as img:
            return _MEDIA_TYPE_BY_PIL_FORMAT.get(img.format or "", "image/jpeg")
    except Exception:
        return "image/jpeg"


def _strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        # Drop the opening fence (possibly ```json) and the closing ```.
        stripped = stripped.split("\n", 1)[1] if "\n" in stripped else stripped[3:]
        if stripped.rstrip().endswith("```"):
            stripped = stripped.rstrip()[:-3]
    return stripped.strip()


def _parse_response(text: str) -> dict:
    return json.loads(_strip_code_fences(text))


def _normalize(raw: dict, raw_text: str) -> ClassifyResult:
    side = str(raw.get("side", "")).lower()
    if side not in {"front", "back"}:
        side = "front"
    return ClassifyResult(
        player=_nullable_str(raw.get("player")),
        team=_nullable_str(raw.get("team")),
        card_number=_nullable_str(raw.get("card_number")),
        side=side,
        raw_text=raw_text,
    )


def _nullable_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _call_model(
    client: anthropic.Anthropic,
    *,
    model: str,
    image_b64: str,
    media_type: str,
    prompt: str,
) -> str:
    response = client.messages.create(
        model=model,
        max_tokens=MAX_TOKENS,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_b64,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    )
    if not response.content:
        return ""
    block = response.content[0]
    return getattr(block, "text", "") or ""


def classify_card(
    image_bytes: bytes,
    *,
    client: anthropic.Anthropic | None = None,
    model: str = DEFAULT_MODEL,
) -> ClassifyResult:
    """Classify a card image.

    Makes up to two Anthropic calls: an initial attempt, and one retry if the
    first response fails to parse as JSON. Raises `ClassifyError` if both
    attempts fail.
    """
    if not image_bytes:
        raise ValueError("image_bytes is empty")

    ai_client = client or anthropic.Anthropic()
    media_type = _detect_media_type(image_bytes)
    image_b64 = base64.b64encode(image_bytes).decode("ascii")

    first_text = _call_model(
        ai_client,
        model=model,
        image_b64=image_b64,
        media_type=media_type,
        prompt=PROMPT,
    )
    try:
        return _normalize(_parse_response(first_text), first_text)
    except (json.JSONDecodeError, TypeError, AttributeError) as first_err:
        logger.warning("classify: first response parse failed, retrying: %s", first_err)

    retry_text = _call_model(
        ai_client,
        model=model,
        image_b64=image_b64,
        media_type=media_type,
        prompt=PROMPT + RETRY_PROMPT_SUFFIX,
    )
    try:
        return _normalize(_parse_response(retry_text), retry_text)
    except (json.JSONDecodeError, TypeError, AttributeError) as retry_err:
        raise ClassifyError(
            f"model response could not be parsed as JSON after retry: {retry_err}"
        ) from retry_err
