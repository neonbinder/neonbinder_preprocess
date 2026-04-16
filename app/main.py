import hmac
import os
from typing import Annotated

from fastapi import FastAPI, Header, HTTPException, status

app = FastAPI(title="neonbinder-preprocess", version="0.1.0")

INTERNAL_API_KEY_ENV = "INTERNAL_API_KEY"


def _verify_internal_key(x_internal_key: str | None) -> None:
    expected = os.environ.get(INTERNAL_API_KEY_ENV)
    if not expected:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="internal api key not configured",
        )
    if not x_internal_key or not hmac.compare_digest(x_internal_key, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid internal key"
        )


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/process", status_code=status.HTTP_501_NOT_IMPLEMENTED)
def process(x_internal_key: Annotated[str | None, Header()] = None) -> dict[str, str]:
    _verify_internal_key(x_internal_key)
    return {"detail": "not implemented"}
