from datetime import UTC, datetime

from catnet.index_registry import IndexRegistry
from catnet.schema import BranchType, EmbeddingVector, Observation
from catnet.storage import CatNetStore
from opencat3d.web import _identity_indexer


def test_index_registry_build_publish_and_rollback(tmp_path) -> None:
    store = CatNetStore(tmp_path / "store")
    first_identity = store.create_identity(seed="cat-a", first_seen_at=_captured_at(0))
    _add_observation_embedding(store, "obs_a", first_identity.cat_id, [1.0, 0.0, 0.0])
    store.save()

    registry = IndexRegistry(tmp_path / "registry")
    first = registry.build_from_store(store, alias="candidate")
    registry.publish(alias="hot", index_version=first.index_version)

    second_identity = store.create_identity(seed="cat-b", first_seen_at=_captured_at(1))
    _add_observation_embedding(store, "obs_b", second_identity.cat_id, [0.0, 1.0, 0.0])
    store.save()
    second = registry.build_from_store(store, alias="candidate")
    registry.publish(alias="hot", index_version=second.index_version)

    assert registry.resolve_alias("hot").index_version.index_version == second.index_version
    rolled_back = registry.rollback(alias="hot")

    assert rolled_back.index_version == first.index_version
    assert len(registry.state.events) >= 5
    assert (tmp_path / "registry" / "indexes" / f"{first.index_version}.json").exists()


def test_catweb_identity_indexer_prefers_hot_alias_and_falls_back(tmp_path) -> None:
    store_dir = tmp_path / "web" / "catnet"
    store = CatNetStore(store_dir)
    identity = store.create_identity(seed="cat-a", first_seen_at=_captured_at(0))
    embedding = _add_observation_embedding(store, "obs_a", identity.cat_id, [1.0, 0.0, 0.0])
    store.save()
    IndexRegistry(store_dir / "index_registry").build_from_store(store, alias="hot")

    hot_index = _identity_indexer(
        store_dir,
        existing_embeddings=[],
        existing_observations={},
        query_embedding=embedding,
    )
    fallback_index = _identity_indexer(
        tmp_path / "empty_registry",
        existing_embeddings=[embedding],
        existing_observations=store.observations_for_embeddings([embedding]),
        query_embedding=embedding,
    )

    assert hot_index.records[0].cat_id == identity.cat_id
    assert hot_index.index_version.alias == "hot"
    assert fallback_index.records[0].cat_id == identity.cat_id


def test_catweb_identity_indexer_falls_back_when_hot_index_integrity_fails(tmp_path) -> None:
    store_dir = tmp_path / "web" / "catnet"
    store = CatNetStore(store_dir)
    identity = store.create_identity(seed="cat-a", first_seen_at=_captured_at(0))
    embedding = _add_observation_embedding(store, "obs_a", identity.cat_id, [1.0, 0.0, 0.0])
    store.save()
    build = IndexRegistry(store_dir / "index_registry").build_from_store(store, alias="hot")
    index_path = store_dir / "index_registry" / "indexes" / f"{build.index_version}.json"
    index_path.write_text('{"corrupted": true}\n', encoding="utf-8")

    index = _identity_indexer(
        store_dir,
        existing_embeddings=[embedding],
        existing_observations=store.observations_for_embeddings([embedding]),
        query_embedding=embedding,
    )

    assert index.index_version.index_version != build.index_version
    assert index.records[0].cat_id == identity.cat_id


def test_index_build_rejects_empty_or_unmatched_store_without_alias_change(tmp_path) -> None:
    registry = IndexRegistry(tmp_path / "registry")

    try:
        registry.build_from_store(CatNetStore(tmp_path / "empty"), alias="candidate")
    except ValueError as exc:
        assert "no fused embeddings" in str(exc)
    else:
        raise AssertionError("empty index build should fail")

    store = CatNetStore(tmp_path / "store")
    identity = store.create_identity(seed="cat-a", first_seen_at=_captured_at(0))
    _add_observation_embedding(store, "obs_a", identity.cat_id, [1.0, 0.0, 0.0])

    try:
        registry.build_from_store(store, alias="candidate", model_version="missing_model")
    except ValueError as exc:
        assert "missing_model" in str(exc)
    else:
        raise AssertionError("unmatched model filter should fail")

    assert "candidate" not in registry.state.aliases


def test_index_publish_rejects_corrupt_candidate_and_preserves_previous_alias(tmp_path) -> None:
    store = CatNetStore(tmp_path / "store")
    first_identity = store.create_identity(seed="cat-a", first_seen_at=_captured_at(0))
    first_embedding = _add_observation_embedding(
        store,
        "obs_a",
        first_identity.cat_id,
        [1.0, 0.0, 0.0],
    )
    registry = IndexRegistry(tmp_path / "registry")
    first = registry.build_from_store(store, alias="candidate")
    registry.publish(alias="hot", index_version=first.index_version)

    second_identity = store.create_identity(seed="cat-b", first_seen_at=_captured_at(1))
    _add_observation_embedding(store, "obs_b", second_identity.cat_id, [0.0, 1.0, 0.0])
    second = registry.build_from_store(store, alias="candidate")
    index_path = tmp_path / "registry" / "indexes" / f"{second.index_version}.json"
    index_path.write_text('{"corrupted": true}\n', encoding="utf-8")

    try:
        registry.publish(alias="hot", index_version=second.index_version)
    except ValueError as exc:
        assert "not loadable" in str(exc)
    else:
        raise AssertionError("corrupt index publish should fail")

    assert registry.state.aliases["hot"].index_version == first.index_version
    assert first_embedding.emb_id == "emb_obs_a"


def _add_observation_embedding(
    store: CatNetStore,
    obs_id: str,
    cat_id: str,
    vector: list[float],
) -> EmbeddingVector:
    observation = Observation(
        obs_id=obs_id,
        observation_version=f"{obs_id}_v1",
        media_id=f"media_{obs_id}",
        cat_id=cat_id,
        captured_at=_captured_at(len(store.catalog.observations)),
    )
    store.upsert_observation(observation)
    embedding = EmbeddingVector(
        emb_id=f"emb_{obs_id}",
        obs_id=obs_id,
        model_version="model_v1",
        branch_type=BranchType.FUSED,
        dim=3,
        vector=vector,
        norm=1.0,
    )
    store.upsert_embedding(embedding)
    return embedding


def _captured_at(day: int) -> datetime:
    return datetime(2026, 1, 1 + day, 8, tzinfo=UTC)
