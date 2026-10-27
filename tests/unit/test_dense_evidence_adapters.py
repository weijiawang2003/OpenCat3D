import numpy as np
import pytest
from pydantic import ValidationError

from opencat3d.adapters.dense_evidence import (
    DENSE_EVIDENCE_MANIFEST_FILENAME,
    DenseEvidencePayload,
    DenseEvidenceRequest,
    LocalFileDenseEvidenceProvider,
    unavailable_dense_evidence_provider,
)
from opencat3d.errors import AdapterUnavailableError, LicenseGateError
from opencat3d.io.cache import sha256_file
from opencat3d.schemas import (
    DenseEvidenceFrame,
    DenseEvidenceKind,
    DenseEvidenceManifest,
    EvidenceProvenance,
    TensorRef,
    TensorStorageFormat,
    read_json_model,
    write_json_model,
)


def _write_observation_manifest(path) -> None:
    path.write_text('{"schema_version":"1.0","frames":[]}\n', encoding="utf-8")


def test_dense_evidence_manifest_serializes_tensor_refs(tmp_path) -> None:
    depth_path = tmp_path / "depth.npy"
    depth = np.ones((8, 6), dtype=np.float32)
    np.save(depth_path, depth)
    depth_ref = TensorRef(
        path=str(depth_path),
        shape=[8, 6],
        dtype="float32",
        sha256=sha256_file(depth_path),
        storage=TensorStorageFormat.NPY,
        semantic="metric_depth_z",
    )
    frame = DenseEvidenceFrame(frame_id="frame_0001", depth=depth_ref)
    manifest = DenseEvidenceManifest(
        project_id="demo_cat",
        source_backend="unit_test_dense_adapter",
        backend_version="0.0",
        license_id="project",
        evidence_kinds=[DenseEvidenceKind.DEPTH],
        frame_ids=["frame_0001"],
        frames=[frame],
        input_hash="0" * 64,
        config_hash="1" * 64,
    )
    path = tmp_path / "dense_evidence.json"

    write_json_model(manifest, path)
    loaded = read_json_model(DenseEvidenceManifest, path)

    assert loaded.project_id == "demo_cat"
    assert loaded.evidence_kinds == [DenseEvidenceKind.DEPTH]
    assert loaded.frames[0].depth is not None
    assert loaded.frames[0].depth.sha256 == sha256_file(depth_path)


def test_local_file_dense_evidence_provider_imports_npy_payloads(tmp_path) -> None:
    observation_manifest = tmp_path / "observations.json"
    _write_observation_manifest(observation_manifest)
    depth_path = tmp_path / "source_depth.npy"
    tracks_path = tmp_path / "source_tracks.npy"
    np.save(depth_path, np.ones((8, 6), dtype=np.float32))
    np.save(tracks_path, np.zeros((2, 4, 2), dtype=np.float32))
    provider = LocalFileDenseEvidenceProvider(
        payloads=(
            DenseEvidencePayload(
                frame_id="frame_0001",
                kind=DenseEvidenceKind.DEPTH,
                path=depth_path,
                provenance=EvidenceProvenance.OBSERVED,
            ),
            DenseEvidencePayload(
                frame_id="frame_0001",
                kind=DenseEvidenceKind.TRACKS,
                path=tracks_path,
                provenance=EvidenceProvenance.INFERRED,
            ),
        )
    )
    request = DenseEvidenceRequest(
        project_id="demo_cat",
        project_dir=tmp_path,
        observation_manifest_path=observation_manifest,
        output_dir=tmp_path / "dense",
        frame_ids=("frame_0001",),
        evidence_kinds=(DenseEvidenceKind.DEPTH, DenseEvidenceKind.TRACKS),
        config={"deterministic": True},
    )

    manifest = provider.estimate(request)
    manifest_path = request.output_dir / DENSE_EVIDENCE_MANIFEST_FILENAME
    loaded = read_json_model(DenseEvidenceManifest, manifest_path)

    assert manifest == loaded
    assert manifest.source_backend == "opencat3d/local_file_dense_evidence"
    assert manifest.frame_ids == ["frame_0001"]
    assert manifest.evidence_kinds == [DenseEvidenceKind.DEPTH, DenseEvidenceKind.TRACKS]
    assert manifest.generated_evidence_fraction == 0.0
    assert manifest.frames[0].provenance == "unknown"
    assert manifest.frames[0].depth is not None
    assert manifest.frames[0].tracks_xy is not None
    assert manifest.frames[0].depth.path.endswith("payloads/frame_0001/depth.npy")
    assert manifest.frames[0].depth.sha256 == sha256_file(
        request.output_dir / "payloads/frame_0001/depth.npy"
    )
    assert manifest.input_hash is not None and len(manifest.input_hash) == 64
    assert manifest.config_hash is not None and len(manifest.config_hash) == 64
    assert "fixed topology" in manifest.notes[0]


def test_local_file_dense_evidence_provider_requires_requested_payloads(tmp_path) -> None:
    observation_manifest = tmp_path / "observations.json"
    _write_observation_manifest(observation_manifest)
    depth_path = tmp_path / "source_depth.npy"
    np.save(depth_path, np.ones((8, 6), dtype=np.float32))
    provider = LocalFileDenseEvidenceProvider(
        payloads=(
            DenseEvidencePayload(
                frame_id="frame_0001",
                kind=DenseEvidenceKind.DEPTH,
                path=depth_path,
            ),
        )
    )
    request = DenseEvidenceRequest(
        project_id="demo_cat",
        project_dir=tmp_path,
        observation_manifest_path=observation_manifest,
        output_dir=tmp_path / "dense",
        frame_ids=("frame_0001",),
        evidence_kinds=(DenseEvidenceKind.DEPTH, DenseEvidenceKind.NORMAL),
    )

    with pytest.raises(ValueError, match="missing dense evidence payload"):
        provider.estimate(request)


def test_local_file_dense_evidence_provider_validates_tensor_shape(tmp_path) -> None:
    observation_manifest = tmp_path / "observations.json"
    _write_observation_manifest(observation_manifest)
    depth_path = tmp_path / "bad_depth.npy"
    np.save(depth_path, np.ones((8, 6, 1), dtype=np.float32))
    provider = LocalFileDenseEvidenceProvider(
        payloads=(
            DenseEvidencePayload(
                frame_id="frame_0001",
                kind=DenseEvidenceKind.DEPTH,
                path=depth_path,
            ),
        )
    )
    request = DenseEvidenceRequest(
        project_id="demo_cat",
        project_dir=tmp_path,
        observation_manifest_path=observation_manifest,
        output_dir=tmp_path / "dense",
        frame_ids=("frame_0001",),
        evidence_kinds=(DenseEvidenceKind.DEPTH,),
    )

    with pytest.raises(ValueError, match="depth evidence must have shape"):
        provider.estimate(request)


def test_dense_evidence_frame_requires_payload() -> None:
    with pytest.raises(ValidationError):
        DenseEvidenceFrame(frame_id="frame_0001")


def test_dense_evidence_manifest_requires_frame_id_order(tmp_path) -> None:
    tensor_path = tmp_path / "tracks.npy"
    np.save(tensor_path, np.zeros((2, 4, 2), dtype=np.float32))
    ref = TensorRef(
        path=str(tensor_path),
        shape=[2, 4, 2],
        dtype="float32",
        sha256=sha256_file(tensor_path),
        storage=TensorStorageFormat.NPY,
        semantic="tracks_xy",
    )

    with pytest.raises(ValidationError):
        DenseEvidenceManifest(
            project_id="demo_cat",
            source_backend="unit_test_dense_adapter",
            evidence_kinds=[DenseEvidenceKind.TRACKS],
            frame_ids=["frame_0002"],
            frames=[DenseEvidenceFrame(frame_id="frame_0001", tracks_xy=ref)],
        )


def test_unavailable_dense_evidence_provider_fails_closed(tmp_path) -> None:
    provider = unavailable_dense_evidence_provider("dust3r_dense_geometry", profile="research")
    request = DenseEvidenceRequest(
        project_id="demo_cat",
        project_dir=tmp_path,
        observation_manifest_path=tmp_path / "manifest.json",
        output_dir=tmp_path / "dense",
        frame_ids=("frame_0001",),
        evidence_kinds=(DenseEvidenceKind.DEPTH,),
    )

    assert provider.info().name == "dust3r_dense_geometry"
    with pytest.raises(AdapterUnavailableError, match="optional and not installed"):
        provider.estimate(request)


def test_unavailable_dense_evidence_provider_keeps_license_gate() -> None:
    provider = unavailable_dense_evidence_provider("dust3r_dense_geometry", profile="product")

    with pytest.raises(LicenseGateError):
        provider.info()
