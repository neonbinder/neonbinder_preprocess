# neonbinder_preprocess

Python FastAPI image-preprocessing service. Moves SAM crop, Vision OCR, and
Anthropic classify off local developer machines and onto a shared Cloud Run
endpoint consumed by the `script-frontend` watcher, `neonbinder_web`, and
`NeonBinderApp`.

See `../image-processing.md` in the neonbinder wrapper for the full design.

## Endpoints

- `GET /health` — liveness probe, returns `{"status":"ok"}`.
- `POST /process` — image preprocessing in three modes (see below).

Auth: all non-health endpoints require the `x-internal-key` header matching
the `INTERNAL_API_KEY` env var (sourced from Secret Manager in Cloud Run).

### `POST /process` modes

Accepts two optional multipart file fields: `image` (the original photo)
and `precropped` (a client-side crop of the card). At least one is
required. The response shape is the same `ProcessResponse` for all three
modes — the mode only affects which work the server performs.

| Mode | `image` | `precropped` | What runs | When to use |
|---|:-:|:-:|---|---|
| **image-only** | yes | — | Full crop cascade (PIL trim → SAM → Haiku bbox → passthrough) on the original. | Callers with no client-side crop capability. |
| **image + precropped** | yes | yes | Tries the crop first; if rejected, falls back to the full cascade on the original. | Callers that can guess a crop but want a server-side safety net. |
| **crop-only** | — | yes | Validates the crop; on pass, runs orient + classify on it. On reject, returns 422 so the caller retries with the original. | Bandwidth-constrained callers whose client-side crops are usually good. Skips the 22 MB-per-image upload entirely when the crop passes. |

On success (all modes) you get a `200` with `ProcessResponse`. Crop-only
mode is the only mode that can return a business-logic 422.

#### 422 — crop-only validation failed

```json
HTTP/1.1 422 Unprocessable Entity
{
  "error_code": "CROP_VALIDATION_FAILED",
  "reason": "aspect 0.857 off by 20%",
  "retry_with_original": true
}
```

`reason` is one of (non-exhaustive, mirrors `ValidationResult.reason`):
`too small WxH`, `aspect … off by N%`, `near-uniform (stddev …)`, or the
special `insufficient_text` when Vision finds no text on the crop. Clients
should treat `retry_with_original: true` as the directive and re-issue the
request with the original image attached as `image`.

#### 400 — missing both fields

```json
HTTP/1.1 400 Bad Request
{
  "error_code": "MISSING_IMAGE",
  "detail": "at least one of image or precropped is required"
}
```

### Client telemetry (for adopters of crop-only mode)

The server emits per-request structured logs with `mode ∈ {image_only,
image_and_crop, crop_only}` and, on crop-only rejection, the rejection
reason. This is the authoritative signal (no sampling, covers all
callers including the script-frontend CLI which does not run PostHog).

Web/mobile callers adopting crop-only mode should additionally emit the
following events to PostHog so the client-observed upload time (the
dominant cost this optimization attacks) is measurable:

- `preprocess_request_started` — props: `mode`, `original_bytes` (if
  attached), `crop_bytes` (if attached).
- `preprocess_request_completed` — props: `duration_ms` (client
  wall-clock), `http_status`, `cropped_source`.
- `preprocess_crop_rejected` — props: `reason`, `retrying_with_original`.
- `preprocess_retry_completed` — props: `duration_ms`.

Gate the rollout on the PostHog feature flag
`preprocess-crop-only-enabled` so adoption can be ramped or halted
without a code deploy.

## Local dev

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
INTERNAL_API_KEY=dev uvicorn app.main:app --reload --port 8080
pytest
ruff check .
```

## Deploy

Deploy jobs are **not yet wired up**. Once `neonbinder_browser` and
`neonbinder_terraform` finish converting to trunk-based development with
per-PR preview revisions, this repo's workflow will mirror the browser
service's final shape (per-PR preview deploys to `neonbinder-dev`,
merge-to-main deploys to `neonbinder` with a traffic-shift gate on smoke).

The current `.github/workflows/preprocess-deploy.yml` only runs lint + unit
tests on PRs and pushes. See `../neonbinder/we-have-a-new-kind-spindle.md`
(Claude Code plan file) for the full intended delivery model.
