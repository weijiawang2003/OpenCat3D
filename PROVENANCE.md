# Source and export provenance

This snapshot was selected from an existing local OpenCat3D working project on 2026-10-26. The original working tree was not modified and had no commits to preserve. The snapshot does not establish individual line-by-line authorship or contributor history.

## Export changes

- Added this README, provenance notes, and freshly generated synthetic example output.
- Fixed wheel packaging of `models/registry.yaml` and added an `importlib.resources` fallback to the registry loader.
- Retained reconstruction/CatNet source, tests, configs, model registry, dependency metadata and original rights/third-party notices.
- Retained `scripts/improve_raw_job.py`, required by the local interface and tests.
- Excluded captures, input EXIF, original runtime outputs, downloaded weights, environments, caches, Git metadata, operational agent instructions, task state, and the autonomous development loop.

`source_manifest.json` records hashes and which files differ from the source workspace. Runtime/resource adapters remain optional; model registry entries are not redistributed weights.

## Verification

41 focused core tests passed. A clean editable install in a fresh Python 3.11 environment passed 23 fast checks and the synthetic command. A wheel was built and its packaged config/registry resources loaded. All 395 bundled tests collected successfully; the full suite was not executed for this release. See the README for the verified command and synthetic metrics.
