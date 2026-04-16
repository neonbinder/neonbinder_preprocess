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

GitHub Actions (`.github/workflows/preprocess-deploy.yml`) deploys on push:

- `develop` → `neonbinder-dev-io`
- `main` → `neonbinder-484017`

Both deploys tag the image in the shared Artifact Registry as
`gcr.io/$GCP_PROJECT/neonbinder-preprocess:$SHA` and update the Cloud Run
service via `google-github-actions/deploy-cloudrun@v2`. Runtime config
(CPU, memory, secrets, SA) is managed by `../neonbinder_terraform/`; CI
only updates the image tag.

## First-deploy ordering

Terraform needs a `:latest` image to reference on first apply. Bootstrap
once manually per GCP project:

```bash
gcloud auth configure-docker --quiet
docker build -t gcr.io/$GCP_PROJECT/neonbinder-preprocess:latest .
docker push gcr.io/$GCP_PROJECT/neonbinder-preprocess:latest
```

Then run Terraform. Thereafter CI handles all image updates.
