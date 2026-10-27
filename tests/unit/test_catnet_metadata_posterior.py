from datetime import UTC, datetime, timedelta

from catnet.matcher.decision import OpenSetDecider, OpenSetDecisionConfig
from catnet.matcher.faiss_index import SearchCandidate
from catnet.metadata import MetadataPosteriorScorer
from catnet.schema import ModelVersion, Observation


def test_metadata_posterior_rewards_same_context_and_penalizes_conflicts() -> None:
    base = _observation(
        "obs_a",
        species="cat",
        site_id="yard",
        camera_id="front",
        captured_at=datetime(2026, 1, 1, 8, tzinfo=UTC),
        uploader="local",
        geohash_public="wtw3s",
    )
    same_context = _observation(
        "obs_b",
        species="cat",
        site_id="yard",
        camera_id="front",
        captured_at=datetime(2026, 1, 8, 9, tzinfo=UTC),
        uploader="local",
        geohash_public="wtw3s",
    )
    conflicting = _observation(
        "obs_c",
        species="dog",
        site_id="other",
        camera_id="back",
        captured_at=datetime(2026, 1, 8, 23, tzinfo=UTC),
        uploader="other",
        geohash_public="abcd1",
    )

    scorer = MetadataPosteriorScorer()
    same_score = scorer.score(base, same_context)
    conflict_score = scorer.score(base, conflicting)

    assert same_score.score > 0.90
    assert same_score.evidence["site"] == "same"
    assert same_score.evidence["camera"] == "same"
    assert conflict_score.score < 0.45
    assert conflict_score.evidence["species"] == "different"
    assert conflict_score.evidence["site"] == "different"


def test_open_set_decider_uses_metadata_posterior_for_ambiguous_context() -> None:
    query = _observation(
        "obs_query",
        species="cat",
        site_id="yard",
        camera_id="front",
        captured_at=datetime(2026, 1, 2, 8, tzinfo=UTC),
        uploader="local",
    )
    same_context = query.model_copy(
        update={
            "obs_id": "obs_same",
            "observation_version": "obs_same_v1",
            "cat_id": "cat_a",
            "captured_at": query.captured_at - timedelta(days=1),
        }
    )
    conflicting = _observation(
        "obs_conflict",
        species="dog",
        site_id="distant",
        camera_id="back",
        captured_at=datetime(2026, 1, 2, 23, tzinfo=UTC),
        uploader="other",
        cat_id="cat_a",
    )
    model_version = ModelVersion(
        model_version="catnet-test",
        backbone="test",
        train_dataset_snapshot="unit",
        calibration_version="test",
        embedding_dim=4,
    )
    decider = OpenSetDecider(
        model_version=model_version,
        index_version="idx_test",
        config=OpenSetDecisionConfig(match_threshold=0.86, review_threshold=0.76),
    )
    candidate = SearchCandidate(
        record_id="rec_known",
        obs_id="obs_known",
        embedding_id="emb_known",
        cat_id="cat_a",
        score=0.90,
        metadata={"global_score": 0.90},
    )

    accepted = decider.decide(
        [candidate],
        query_observation=query,
        observations_by_id={"obs_known": same_context},
    )
    needs_review = decider.decide(
        [candidate],
        query_observation=query,
        observations_by_id={"obs_known": conflicting},
    )

    assert accepted.decision.value == "match_existing"
    assert accepted.metadata_score is not None and accepted.metadata_score > 0.90
    assert accepted.explanation["metadata"]["site"] == "same"
    assert needs_review.decision.value == "needs_review"
    assert needs_review.metadata_score is not None and needs_review.metadata_score < 0.45
    assert needs_review.explanation["metadata"]["species"] == "different"


def _observation(
    obs_id: str,
    *,
    species: str,
    site_id: str,
    camera_id: str,
    captured_at: datetime,
    uploader: str,
    geohash_public: str | None = None,
    cat_id: str | None = None,
) -> Observation:
    return Observation(
        obs_id=obs_id,
        observation_version=f"{obs_id}_v1",
        media_id=f"media_{obs_id}",
        cat_id=cat_id,
        species=species,
        captured_at=captured_at,
        site_id=site_id,
        camera_id=camera_id,
        geohash_public=geohash_public,
        quality_score=0.92,
        metadata={"uploader": uploader},
    )
