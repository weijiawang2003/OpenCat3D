# Model Registry

`registry.yaml` records external model, adapter, and weight license metadata. Core code
must not import heavyweight adapter dependencies or download weights at import time.

## CatNet Re-ID Backbones

CatNet keeps heavyweight Re-ID backbones optional. Install them only when training
or extracting external features:

```bash
uv sync --extra reid
```

Available adapter names:

- `deterministic`: local color/texture/metadata baseline used by tests.
- `megadescriptor-t-224`: `timm` adapter for `hf-hub:BVRA/MegaDescriptor-T-224`.
- `megadescriptor-l-384`: `timm` adapter for `hf-hub:BVRA/MegaDescriptor-L-384`.
- `miewid-msv3`: `transformers` adapter for `conservationxlabs/miewid-msv3`.
- `hybrid:megadescriptor-t-224+deterministic`: recommended first CatNet recipe for
  cat data because it combines a pretrained animal Re-ID feature backbone with the
  local coat/color branch.

MegaDescriptor model cards list `CC-BY-NC-4.0`, so keep it in
research/non-commercial profiles unless licensing is resolved.

## Research Integrations In Code

- WildlifeDatasets/PetFace-style inputs are normalized through CatNet manifest
  adapters, then trained with `opencat3d catnet train`.
- MegaDescriptor and MiewID are optional backbone adapters behind
  `uv sync --extra reid`; no weights are downloaded at import time.
- WildFusion/RAPID-style local evidence is represented by
  `LocalTextureReranker`, which reranks top-k global candidates and writes
  `MatchEdge.local_score` plus `local_rerank` evidence.
- MetaWild-style metadata evidence is represented by `MetadataPosteriorScorer`,
  which scores species/site/camera/geohash/uploader/encounter/time-of-day
  compatibility and writes `MatchEdge.metadata_score`.
- AAS-style active learning is represented by
  `opencat3d catnet review-queue`, which ranks low-margin, score-disagreement,
  and needs-review edges for human labeling.
- Index deployment is represented by `opencat3d catnet index build/publish/rollback/status`.
  It stores immutable JSON vector indexes and hot/main/candidate aliases, with CatWeb
  resolving the hot/main alias before falling back to an in-memory exact index. Alias
  loads validate manifest paths and SHA-256 before use, so corrupted or stale hot indexes
  fall back instead of breaking local matching.
- Reviewer feedback is represented by `opencat3d catnet feedback add/import/apply`.
  Must-link/cannot-link/same/different/unsure labels are durable constraints; apply is
  proposal/audit-first unless explicitly requested. Missing observation or identity
  references are reported as invalid constraints, and repeated apply runs do not
  duplicate identity events.
- Repeatable comparison runs are represented by `opencat3d catnet benchmark`.
  Missing external manifests write an explicit blocked report, while deterministic and
  hybrid deterministic backbones remain CI-safe. Corrupt image inputs are contained to
  per-backbone blocked reports instead of aborting the whole benchmark.
- Graph cleanup is represented by `opencat3d catnet consolidate`, which emits merge,
  split, and blocked proposals with source edge ids and feedback constraint ids.
  Repeated `--apply` runs are idempotent by proposal id.
- CatWeb exposes `/identity/review` and `/identity/review/<task_id>` for human
  Same Cat / Different Cat / Unsure writeback. Review media is served only from the
  configured CatWeb work directory; external catalog paths require explicit ingestion.
- 3D artifacts attach after identity through `opencat3d catnet attach-3d` and
  `opencat3d catnet identity-assets`; CatNet matching does not depend on 3D assets.
  Source observations must already belong to the target identity.

Next heavy integrations should add a LoFTR/LightGlue local matcher behind the
same rerank interface, run the benchmark harness on mounted CatIndividualImages/PetFace
manifests, and resolve external backbone licensing before commercial use.
