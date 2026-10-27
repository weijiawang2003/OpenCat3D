"""Identity-owned bridge records for OpenCat3D reconstruction artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from catnet.schema import Identity3DAsset, IdentityEventType, stable_id
from catnet.storage import CatNetStore


def attach_3d_asset(
    store: CatNetStore,
    *,
    cat_id: str,
    project_dir: Path,
    source_obs_ids: list[str] | None = None,
    reconstruction_version: str = "opencat3d-v1",
    mesh_path: Path | None = None,
    glb_path: Path | None = None,
    texture_path: Path | None = None,
    quality_report_path: Path | None = None,
) -> Identity3DAsset:
    if cat_id not in store.catalog.identities:
        raise ValueError(f"unknown CatNet identity: {cat_id}")
    resolved_project_dir = Path(project_dir).resolve()
    if not resolved_project_dir.is_dir():
        raise FileNotFoundError(resolved_project_dir)

    resolved_mesh = _resolve_optional_path(mesh_path) or _first_existing(
        resolved_project_dir,
        ["outputs/avatar/cat_canonical.obj"],
    )
    resolved_glb = _resolve_optional_path(glb_path) or _first_existing(
        resolved_project_dir,
        ["outputs/avatar/cat_avatar.glb"],
    )
    resolved_texture = _resolve_optional_path(texture_path) or _first_existing(
        resolved_project_dir,
        [
            "outputs/textures/texture_manifest.json",
            "outputs/textures/basecolor.png",
        ],
    )
    resolved_quality = _resolve_optional_path(quality_report_path) or _first_existing(
        resolved_project_dir,
        [
            "outputs/quality/raw_evaluation.json",
            "outputs/quality/synthetic_evaluation.json",
            "outputs/quality/raw_pipeline_summary.json",
            "outputs/quality/synthetic_pipeline_summary.json",
            "outputs/quality/validation.json",
        ],
    )
    if not any([resolved_mesh, resolved_glb, resolved_texture, resolved_quality]):
        raise FileNotFoundError(
            "project does not contain recognized OpenCat3D mesh, GLB, texture, or quality files"
        )

    obs_ids = (
        source_obs_ids
        if source_obs_ids is not None
        else _identity_observation_ids(store, cat_id)
    )
    missing_obs_ids = [obs_id for obs_id in obs_ids if obs_id not in store.catalog.observations]
    if missing_obs_ids:
        raise ValueError(f"unknown source observations: {', '.join(sorted(missing_obs_ids))}")
    mismatched_obs_ids = [
        obs_id
        for obs_id in obs_ids
        if store.catalog.observations[obs_id].cat_id != cat_id
    ]
    if mismatched_obs_ids:
        raise ValueError(
            "source observations must be assigned to the target identity: "
            + ", ".join(sorted(mismatched_obs_ids))
        )

    asset = Identity3DAsset(
        asset_id=stable_id(
            "asset3d",
            cat_id,
            resolved_project_dir,
            reconstruction_version,
            resolved_mesh or "",
            resolved_glb or "",
        ),
        cat_id=cat_id,
        source_obs_ids=sorted(obs_ids),
        project_dir=str(resolved_project_dir),
        reconstruction_version=reconstruction_version,
        mesh_path=str(resolved_mesh) if resolved_mesh else None,
        glb_path=str(resolved_glb) if resolved_glb else None,
        texture_path=str(resolved_texture) if resolved_texture else None,
        quality_report_path=str(resolved_quality) if resolved_quality else None,
        metadata={
            "bridge_version": "catnet-3d-bridge-v1",
            "dependency_direction": "catnet_identity_to_opencat3d_artifact",
        },
    )
    store.upsert_3d_asset(asset)
    identity = store.catalog.identities[cat_id]
    store.record_identity_event(
        cat_id=cat_id,
        event_type=IdentityEventType.UPDATED,
        reason="attached OpenCat3D asset",
        previous_identity_version=identity.identity_version,
        payload=asset.model_dump(mode="json"),
    )
    return asset


def list_identity_assets(store: CatNetStore, *, cat_id: str) -> list[Identity3DAsset]:
    if cat_id not in store.catalog.identities:
        raise ValueError(f"unknown CatNet identity: {cat_id}")
    return sorted(
        [
            asset
            for asset in store.catalog.identity_3d_assets.values()
            if asset.cat_id == cat_id
        ],
        key=lambda asset: (asset.created_at, asset.asset_id),
    )


def write_identity_assets(assets: list[Identity3DAsset], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "1.0",
        "asset_count": len(assets),
        "assets": [asset.model_dump(mode="json") for asset in assets],
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


def _identity_observation_ids(store: CatNetStore, cat_id: str) -> list[str]:
    return [
        observation.obs_id
        for observation in store.catalog.observations.values()
        if observation.cat_id == cat_id
    ]


def _resolve_optional_path(path: Path | None) -> Path | None:
    if path is None:
        return None
    resolved = path.resolve()
    if not resolved.exists():
        raise FileNotFoundError(resolved)
    return resolved


def _first_existing(project_dir: Path, relative_paths: list[str]) -> Path | None:
    for relative_path in relative_paths:
        candidate = (project_dir / relative_path).resolve()
        if candidate.exists():
            return candidate
    return None
