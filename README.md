# neonbinder_preprocess

Python FastAPI image-preprocessing service. Moves SAM crop, Vision OCR, and
Anthropic classify off local developer machines and onto a shared Cloud Run
endpoint consumed by the `script-frontend` watcher, `neonbinder_web`, and
`NeonBinderApp`.

See `../image-processing.md` in the neonbinder wrapper for the full design.

## Endpoints

- `GET /healthz` — liveness probe, returns `{"status":"ok"}`.
- `POST /process` — already-cropped image → orient + classify (slice 1, in progress).
- `POST /crop-and-process` — raw photo → SAM crop → orient + classify (slice 2).

Auth: all non-health endpoints require the `x-internal-key` header matching
the `INTERNAL_API_KEY` env var (sourced from Secret Manager in Cloud Run).

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
