from datetime import UTC, datetime

import numpy as np
from PIL import Image

from catnet.matcher.decision import OpenSetDecider, OpenSetDecisionConfig
from catnet.matcher.faiss_index import SearchCandidate
from catnet.matcher.rerank import LocalRerankConfig, LocalTextureReranker
from catnet.schema import ModelVersion, Observation


def test_local_texture_reranker_promotes_spatially_matching_pattern(tmp_path) -> None:
    query = tmp_path / "query.png"
    same = tmp_path / "same.png"
    shifted = tmp_path / "shifted.png"
    _pattern_image(query, square=(8, 8, 30, 30))
    _pattern_image(same, square=(8, 8, 30, 30))
    _pattern_image(shifted, square=(66, 66, 88, 88))

    candidates = [
        _candidate("obs_shifted", score=0.91),
        _candidate("obs_same", score=0.90),
    ]
    result = LocalTextureReranker(
        LocalRerankConfig(global_weight=0.60, local_weight=0.40)
    ).rerank(
        query_image_path=query,
        candidates=candidates,
        candidate_image_paths={"obs_same": same, "obs_shifted": shifted},
    )

    assert [candidate.obs_id for candidate in result.candidates] == ["obs_same", "obs_shifted"]
    assert result.candidates[0].metadata["local_score"] > result.candidates[1].metadata[
        "local_score"
    ]
    assert result.candidates[0].metadata["local_rerank"]["mutual_match_count"] > 0


def test_open_set_decider_records_local_rerank_evidence(tmp_path) -> None:
    query = tmp_path / "query.png"
    known = tmp_path / "known.png"
    _pattern_image(query, square=(8, 8, 30, 30))
    _pattern_image(known, square=(8, 8, 30, 30))
    candidates = LocalTextureReranker().rerank(
        query_image_path=query,
        candidates=[_candidate("obs_known", score=0.90, cat_id="cat_a")],
        candidate_image_paths={"obs_known": known},
    ).candidates
    query_observation = _observation("obs_query", captured_at=datetime(2026, 1, 2, tzinfo=UTC))
    known_observation = _observation("obs_known", captured_at=datetime(2026, 1, 1, tzinfo=UTC))
    model_version = ModelVersion(
        model_version="catnet-test",
        backbone="test",
        train_dataset_snapshot="unit",
        calibration_version="test",
        embedding_dim=4,
    )

    decision = OpenSetDecider(
        model_version=model_version,
        index_version="idx_test",
        config=OpenSetDecisionConfig(match_threshold=0.80, review_threshold=0.70),
    ).decide(
        candidates,
        query_observation=query_observation,
        observations_by_id={"obs_known": known_observation},
    )

    assert decision.decision.value == "match_existing"
    assert decision.local_score is not None
    assert decision.explanation["top_global_score"] == 0.9
    assert decision.explanation["top_local_score"] == round(decision.local_score, 6)
    assert decision.explanation["local_rerank"]["status"] == "ok"


def _candidate(obs_id: str, *, score: float, cat_id: str | None = None) -> SearchCandidate:
    return SearchCandidate(
        record_id=f"rec_{obs_id}",
        obs_id=obs_id,
        embedding_id=f"emb_{obs_id}",
        cat_id=cat_id,
        score=score,
        metadata={"species": "cat"},
    )


def _observation(obs_id: str, *, captured_at: datetime) -> Observation:
    return Observation(
        obs_id=obs_id,
        observation_version=f"{obs_id}_v1",
        media_id=f"media_{obs_id}",
        cat_id="cat_a" if obs_id == "obs_known" else None,
        captured_at=captured_at,
        site_id="yard",
        camera_id="front",
        age_bin="adult",
    )


def _pattern_image(path, *, square: tuple[int, int, int, int]) -> None:
    image = np.full((96, 96, 3), 235, dtype=np.uint8)
    x0, y0, x1, y1 = square
    image[y0:y1, x0:x1, :] = np.asarray([35, 35, 35], dtype=np.uint8)
    image[42:54, 20:76, 1] = 120
    Image.fromarray(image).save(path)
