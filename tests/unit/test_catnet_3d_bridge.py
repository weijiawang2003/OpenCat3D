from datetime import UTC, datetime

from catnet.assets3d import attach_3d_asset, list_identity_assets, write_identity_assets
from catnet.schema import Observation
from catnet.storage import CatNetStore


def test_attach_3d_records_identity_owned_opencat3d_assets(tmp_path) -> None:
    store = CatNetStore(tmp_path / "catnet")
    identity = store.create_identity(seed="cat-a", first_seen_at=datetime(2026, 1, 1, tzinfo=UTC))
    store.upsert_observation(
        Observation(
            obs_id="obs_a",
            observation_version="obs_a_v1",
            media_id="media_a",
            cat_id=identity.cat_id,
        )
    )
    project_dir = tmp_path / "project"
    _write_project_artifacts(project_dir)

    asset = attach_3d_asset(store, cat_id=identity.cat_id, project_dir=project_dir)
    store.save()
    assets = list_identity_assets(CatNetStore(tmp_path / "catnet"), cat_id=identity.cat_id)
    output = write_identity_assets(assets, tmp_path / "assets.json")

    assert asset.cat_id == identity.cat_id
    assert asset.source_obs_ids == ["obs_a"]
    assert asset.mesh_path.endswith("outputs/avatar/cat_canonical.obj")
    assert asset.glb_path.endswith("outputs/avatar/cat_avatar.glb")
    assert asset.texture_path.endswith("outputs/textures/texture_manifest.json")
    assert asset.quality_report_path.endswith("outputs/quality/raw_evaluation.json")
    assert assets[0].asset_id == asset.asset_id
    assert output.exists()


def test_attach_3d_requires_existing_catnet_identity(tmp_path) -> None:
    store = CatNetStore(tmp_path / "catnet")
    project_dir = tmp_path / "project"
    _write_project_artifacts(project_dir)

    try:
        attach_3d_asset(store, cat_id="cat_missing", project_dir=project_dir)
    except ValueError as exc:
        assert "unknown CatNet identity" in str(exc)
    else:
        raise AssertionError("expected missing identity to fail")


def test_attach_3d_rejects_source_observation_from_other_identity(tmp_path) -> None:
    store = CatNetStore(tmp_path / "catnet")
    first = store.create_identity(seed="cat-a", first_seen_at=datetime(2026, 1, 1, tzinfo=UTC))
    second = store.create_identity(seed="cat-b", first_seen_at=datetime(2026, 1, 2, tzinfo=UTC))
    store.upsert_observation(
        Observation(
            obs_id="obs_b",
            observation_version="obs_b_v1",
            media_id="media_b",
            cat_id=second.cat_id,
        )
    )
    project_dir = tmp_path / "project"
    _write_project_artifacts(project_dir)

    try:
        attach_3d_asset(
            store,
            cat_id=first.cat_id,
            project_dir=project_dir,
            source_obs_ids=["obs_b"],
        )
    except ValueError as exc:
        assert "source observations must be assigned" in str(exc)
    else:
        raise AssertionError("expected mismatched source observation to fail")


def test_identity_assets_requires_known_identity(tmp_path) -> None:
    store = CatNetStore(tmp_path / "catnet")

    try:
        list_identity_assets(store, cat_id="cat_missing")
    except ValueError as exc:
        assert "unknown CatNet identity" in str(exc)
    else:
        raise AssertionError("expected unknown identity lookup to fail")


def _write_project_artifacts(project_dir) -> None:
    for relative_path in [
        "outputs/avatar/cat_canonical.obj",
        "outputs/avatar/cat_avatar.glb",
        "outputs/textures/texture_manifest.json",
        "outputs/quality/raw_evaluation.json",
    ]:
        path = project_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
