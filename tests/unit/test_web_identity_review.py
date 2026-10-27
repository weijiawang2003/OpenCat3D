import hashlib
from datetime import UTC, datetime
from http.client import HTTPConnection
from io import BytesIO
from threading import Thread

import numpy as np
from PIL import Image

from catnet.active_learning import build_ambiguity_queue
from catnet.schema import DecisionType, FeedbackConstraintType, MatchEdge, MediaAsset, Observation
from catnet.storage import CatNetStore
from opencat3d import web
from opencat3d.web import render_identity_review_task


def test_review_page_renders_task_images_and_scores(tmp_path) -> None:
    task_id = _write_review_fixture(tmp_path)

    html = render_identity_review_task(tmp_path, task_id=task_id)

    assert "/identity/media/media_query" in html
    assert "/identity/media/media_candidate" in html
    assert "0.810" in html
    assert "Same Cat" in html
    assert "Different Cat" in html


def test_feedback_post_writes_constraint(tmp_path) -> None:
    task_id = _write_review_fixture(tmp_path)
    server = web.make_web_server(host="127.0.0.1", port=0, work_dir=tmp_path)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        connection = HTTPConnection("127.0.0.1", server.server_address[1], timeout=5)
        connection.request(
            "POST",
            f"/identity/review/{task_id}",
            body=b"feedback_action=same",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        response = connection.getresponse()
        response.read()
        connection.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    store = CatNetStore(tmp_path / "catnet")
    constraints = list(store.catalog.feedback_constraints.values())

    assert response.status == 303
    assert response.headers["Location"] == "/identity/review"
    assert constraints[0].constraint_type == FeedbackConstraintType.MUST_LINK
    assert constraints[0].review_task_id == task_id


def test_feedback_post_invalid_action_returns_400_without_constraint(tmp_path) -> None:
    task_id = _write_review_fixture(tmp_path)
    status = _post_review_status(tmp_path, task_id, b"feedback_action=bad")

    store = CatNetStore(tmp_path / "catnet")

    assert status == 400
    assert not store.catalog.feedback_constraints


def test_feedback_post_same_without_candidate_returns_400_without_constraint(tmp_path) -> None:
    task_id = _write_single_observation_review_fixture(tmp_path)
    status = _post_review_status(tmp_path, task_id, b"feedback_action=same")

    store = CatNetStore(tmp_path / "catnet")

    assert status == 400
    assert not store.catalog.feedback_constraints


def test_review_routes_reject_missing_and_unsafe_tasks(tmp_path) -> None:
    _write_review_fixture(tmp_path)
    server = web.make_web_server(host="127.0.0.1", port=0, work_dir=tmp_path)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        missing = _get_status(server, "/identity/review/task_missing")
        unsafe = _get_status(server, "/identity/review/..%2Fsecret")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert missing == 404
    assert unsafe == 404


def test_identity_media_rejects_catalog_paths_outside_work_dir(tmp_path) -> None:
    _write_review_fixture(tmp_path)
    outside_path = tmp_path.parent / "outside_identity_media.png"
    outside_path.write_bytes(_png_bytes(90))
    store = CatNetStore(tmp_path / "catnet")
    media = store.catalog.media_assets["media_query"]
    store.catalog.media_assets["media_query"] = media.model_copy(
        update={"storage_uri": str(outside_path)}
    )
    store.save()
    server = web.make_web_server(host="127.0.0.1", port=0, work_dir=tmp_path)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status = _get_status(server, "/identity/media/media_query")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert status == 404


def _write_review_fixture(root) -> str:
    store = CatNetStore(root / "catnet")
    first = store.create_identity(seed="cat-a", first_seen_at=datetime(2026, 1, 1, tzinfo=UTC))
    second = store.create_identity(seed="cat-b", first_seen_at=datetime(2026, 1, 2, tzinfo=UTC))
    query_path = root / "query.png"
    candidate_path = root / "candidate.png"
    query_bytes = _png_bytes(70)
    candidate_bytes = _png_bytes(80)
    query_path.write_bytes(query_bytes)
    candidate_path.write_bytes(candidate_bytes)
    store.upsert_media(_media("media_query", query_path, query_bytes))
    store.upsert_media(_media("media_candidate", candidate_path, candidate_bytes))
    store.upsert_observation(_obs("obs_query", "media_query", first.cat_id))
    store.upsert_observation(_obs("obs_candidate", "media_candidate", second.cat_id))
    edge = MatchEdge(
        edge_id="edge_review",
        src_obs_id="obs_query",
        dst_obs_id="obs_candidate",
        src_cat_id=first.cat_id,
        dst_cat_id=second.cat_id,
        global_score=0.81,
        local_score=0.62,
        temporal_score=0.74,
        metadata_score=0.58,
        fused_score=0.76,
        decision=DecisionType.NEEDS_REVIEW,
        confidence=0.55,
        model_version="model_v1",
        index_version="idx_v1",
        calibration_version="cal_v1",
        decision_rule_version="rule_v1",
        explanation={"reason": "unit_test"},
    )
    store.add_match_edge(edge)
    store.save()
    return build_ambiguity_queue([edge])[0].task_id


def _write_single_observation_review_fixture(root) -> str:
    store = CatNetStore(root / "catnet")
    identity = store.create_identity(seed="cat-a", first_seen_at=datetime(2026, 1, 1, tzinfo=UTC))
    query_path = root / "query.png"
    query_bytes = _png_bytes(70)
    query_path.write_bytes(query_bytes)
    store.upsert_media(_media("media_query", query_path, query_bytes))
    store.upsert_observation(_obs("obs_query", "media_query", identity.cat_id))
    edge = MatchEdge(
        edge_id="edge_review_single",
        src_obs_id="obs_query",
        dst_obs_id=None,
        src_cat_id=identity.cat_id,
        dst_cat_id=None,
        global_score=0.80,
        local_score=None,
        temporal_score=0.50,
        metadata_score=None,
        fused_score=0.74,
        decision=DecisionType.NEEDS_REVIEW,
        confidence=0.50,
        model_version="model_v1",
        index_version="idx_v1",
        calibration_version="cal_v1",
        decision_rule_version="rule_v1",
        explanation={"reason": "unit_test"},
    )
    store.add_match_edge(edge)
    store.save()
    return build_ambiguity_queue([edge])[0].task_id


def _media(media_id: str, path, content: bytes) -> MediaAsset:
    return MediaAsset(
        media_id=media_id,
        storage_uri=str(path),
        sha256=hashlib.sha256(content).hexdigest(),
        width=8,
        height=8,
        mime_type="image/png",
    )


def _obs(obs_id: str, media_id: str, cat_id: str) -> Observation:
    return Observation(
        obs_id=obs_id,
        observation_version=f"{obs_id}_v1",
        media_id=media_id,
        cat_id=cat_id,
        captured_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


def _png_bytes(value: int) -> bytes:
    image = np.full((8, 8, 3), 255, dtype=np.uint8)
    image[2:6, 2:6, :] = value
    buffer = BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    return buffer.getvalue()


def _get_status(server, path: str) -> int:
    connection = HTTPConnection("127.0.0.1", server.server_address[1], timeout=5)
    connection.request("GET", path)
    response = connection.getresponse()
    response.read()
    connection.close()
    return response.status


def _post_review_status(root, task_id: str, body: bytes) -> int:
    server = web.make_web_server(host="127.0.0.1", port=0, work_dir=root)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        connection = HTTPConnection("127.0.0.1", server.server_address[1], timeout=5)
        connection.request(
            "POST",
            f"/identity/review/{task_id}",
            body=body,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        response = connection.getresponse()
        response.read()
        connection.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    return response.status
