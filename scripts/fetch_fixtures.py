#!/usr/bin/env python3
"""Download fixture images from GCS.

TODO(terraform): wire up once `gs://neonbinder-dev-preprocess-fixtures` exists.
Blocked on the trunk-based conversion of `neonbinder_terraform/` finishing;
creating the bucket requires Terraform per project policy (no out-of-band
GCP resource creation).

Until then, fixture images are obtained manually — see
`tests/fixtures/README.md`. This script exists as a placeholder so the path
forward is discoverable from the repo.

Planned behavior:
    1. Read `tests/fixtures/*.yaml` to discover which images are referenced.
    2. Check which image files are missing locally.
    3. For each missing image, `gsutil cp` from the bucket. Use ADC; the
       developer's Owner creds on neonbinder-dev are sufficient.
    4. Report anything that failed to download with a clear error.
"""

from __future__ import annotations

import sys

FIXTURES_BUCKET: str | None = None  # set once Terraform creates the bucket


def main() -> int:
    if FIXTURES_BUCKET is None:
        print(
            "Fixture bucket not yet provisioned. See tests/fixtures/README.md "
            "for the manual-copy workflow. This script will be wired up once "
            "the Terraform trunk-based conversion lands."
        )
        return 1
    # Implementation deferred until bucket exists.
    return 0


if __name__ == "__main__":
    sys.exit(main())
