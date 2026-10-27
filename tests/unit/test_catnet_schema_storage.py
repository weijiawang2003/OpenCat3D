from datetime import UTC, datetime

from catnet.schema import (
    BranchType,
    EmbeddingVector,
    MediaAsset,
    Observation,
)
from catnet.storage import CatNetStore


def test_catnet_store_round_trips_versioned_records(tmp_path) -> None:
    store = CatNetStore(tmp_path)
    captured_at = datetime(2026, 1, 5, 10, 30, tzinfo=UTC)
    media = MediaAsset(
        media_id="media_a",
        storage_uri="/tmp/cat.png",
        sha256="a" * 64,
        width=32,
        height=24,
        mime_type="image/png",
    )
    observation = Observation(
        obs_id="obs_a",
        observation_version="obv_a",
        media_id=media.media_id,
        captured_at=captured_at,
        site_id="yard",
        camera_id="front",
    )
    identity = store.create_identity(seed="cat-a", first_seen_at=captured_at)

    store.upsert_media(media)
    store.upsert_observation(observation)
    assigned = store.assign_observation(
        observation.obs_id,
        identity.cat_id,
        reason="test assignment",
    )
    store.upsert_embedding(
        EmbeddingVector(
            emb_id="emb_a",
            obs_id=assigned.obs_id,
            model_version="model_v1",
            branch_type=BranchType.FUSED,
            dim=3,
            vector=[1.0, 0.0, 0.0],
            norm=1.0,
        )
    )
    store.save()

    loaded = CatNetStore(tmp_path)

    assert loaded.catalog.media_assets[media.media_id].sha256 == "a" * 64
    assert loaded.catalog.observations[observation.obs_id].cat_id == identity.cat_id
    assert loaded.catalog.identities[identity.cat_id].first_seen_at == captured_at
    assert loaded.list_embeddings(branch_type=BranchType.FUSED)[0].emb_id == "emb_a"
    assert loaded.catalog.identity_versions
