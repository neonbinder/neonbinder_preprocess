# Fixture images

Binary images are stored out-of-band. Only the `.yaml` sidecars are tracked in
git. The sidecars declare expected orient + classify outputs per image; see
`tests/integration/_loader.py` for the schema.

## Getting images

Currently manual — drop the matching image file next to each tracked sidecar:

```
tests/fixtures/
  black-border-back.jpg    ← not in git, drop here
  black-border-back.yaml   ← in git, already here
  landscape.jpg
  landscape.yaml
  white-border-front.jpg
  white-border-front.yaml
```

Ask `@jburich` for a copy of the current fixture set (they live on his Mac
for now).

## Planned (pending Terraform work)

Once `neonbinder_terraform/` is done with the trunk-based conversion, we'll
add a GCS bucket `gs://neonbinder-dev-preprocess-fixtures` and
`scripts/fetch_fixtures.py` will pull missing images from it automatically
before tests run.

## Adding a new fixture

1. Drop the image next to the other fixtures (any of `.jpg`, `.jpeg`, `.png`,
   `.webp`).
2. Run `python scripts/label_fixtures.py --fixture <name>.jpg` to bootstrap a
   sidecar from a real pipeline run.
3. Review the generated sidecar. Loosen `equals:` to `contains:` on
   player/team if the exact string is likely to drift. Delete fields you
   don't want to assert. Card_number stays exact per project policy.
4. Commit the sidecar (image stays local / out-of-band).
5. Run `RUN_INTEGRATION_TESTS=1 pytest tests/integration -v` to confirm the
   sidecar passes.
