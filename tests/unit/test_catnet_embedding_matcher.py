from datetime import UTC, datetime, timedelta
from io import BytesIO

import numpy as np
from PIL import Image

from catnet.embedding.service import EmbeddingService
from catnet.matcher.decision import OpenSetDecider
from catnet.matcher.faiss_index import FaissIndex
from catnet.schema import BranchType, Observation
from catnet.storage import CatNetStore


def test_embedding_service_emits_versioned_multibranch_vectors() -> None:
    service = EmbeddingService()
    observation = _observation("obs_query", captured_at=datetime(2026, 1, 2, tzinfo=UTC))

    embeddings = service.extract(_png_bytes(40), observation=observation, obs_id=observation.obs_id)

    assert {embedding.branch_type for embedding in embeddings} == set(BranchType)
    assert all(
        embedding.model_version == service.model_version.model_version
        for embedding in embeddings
    )
    assert all(embedding.dim == service.model_version.embedding_dim for embedding in embeddings)
    assert all(0.99 <= embedding.norm <= 1.01 for embedding in embeddings)


def test_faiss_index_and_open_set_decider_match_identical_identity(tmp_path) -> None:
    service = EmbeddingService()
    store = CatNetStore(tmp_path)
    first_at = datetime(2026, 1, 1, 8, tzinfo=UTC)
    first_obs = _observation("obs_known", captured_at=first_at)
    identity = store.create_identity(seed="known", first_seen_at=first_at)
    store.upsert_observation(first_obs)
    first_obs = store.assign_observation(first_obs.obs_id, identity.cat_id, reason="seed")
    first_embedding = service.extract_fused(
        _png_bytes(36),
        observation=first_obs,
        obs_id=first_obs.obs_id,
    )
    store.upsert_embedding(first_embedding)

    query_obs = _observation("obs_query", captured_at=first_at + timedelta(days=2))
    query_embedding = service.extract_fused(
        _png_bytes(36),
        observation=query_obs,
        obs_id=query_obs.obs_id,
    )
    existing_embeddings = store.list_embeddings(branch_type=BranchType.FUSED)
    observations = store.observations_for_embeddings(existing_embeddings)
    index = FaissIndex.from_embeddings(existing_embeddings, observations)
    candidates = index.search(query_embedding.vector or [], top_k=5)
    decision = OpenSetDecider(
        model_version=service.model_version,
        index_version=index.index_version.index_version,
    ).decide(candidates, query_observation=query_obs, observations_by_id=observations)

    assert candidates[0].cat_id == identity.cat_id
    assert candidates[0].score > 0.99
    assert decision.decision.value == "match_existing"
    assert decision.cat_id == identity.cat_id
    assert decision.explanation["temporal"]["gap_days"] == 2.0


def test_empty_index_drives_open_set_new_identity_decision() -> None:
    service = EmbeddingService()
    query_obs = _observation("obs_query", captured_at=datetime(2026, 1, 1, tzinfo=UTC))
    index = FaissIndex(
        dim=service.model_version.embedding_dim,
        model_version=service.model_version.model_version,
    )
    decision = OpenSetDecider(
        model_version=service.model_version,
        index_version=index.index_version.index_version,
    ).decide([], query_observation=query_obs, observations_by_id={})

    assert decision.decision.value == "create_new_identity"
    assert decision.confidence == 1.0
    assert decision.explanation["reason"] == "no_candidates"


def _observation(obs_id: str, *, captured_at: datetime) -> Observation:
    return Observation(
        obs_id=obs_id,
        observation_version=f"{obs_id}_v1",
        media_id=f"media_{obs_id}",
        captured_at=captured_at,
        site_id="yard",
        camera_id="front",
        age_bin="adult",
    )


def _png_bytes(value: int) -> bytes:
    image = np.full((16, 16, 3), 255, dtype=np.uint8)
    image[4:12, 4:12, :] = value
    buffer = BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    return buffer.getvalue()
