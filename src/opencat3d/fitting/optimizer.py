"""Staged optimizer skeleton for synthetic camera/root fitting."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from safetensors import safe_open
from safetensors.numpy import save_file

from opencat3d.camera.conventions import project_world_points
from opencat3d.fitting.losses import (
    feature_anchor_projection_loss,
    keypoint_reprojection_loss,
    soft_iou_loss,
)
from opencat3d.fitting.overlays import write_fit_overlays
from opencat3d.schemas import (
    FitReport,
    FrameObservationMeta,
    InitPredictionMeta,
    PerspectiveCameraState,
    TensorRef,
    write_json_model,
)
from opencat3d.synthetic.renderer import render_silhouette
from opencat3d.template.cat_template import CatTemplate, load_cat_template_npz
from opencat3d.template.skeleton import JOINT_NAMES

_CHECKPOINT_VERSION = 1
_CHECKPOINT_AFTER_CAMERA_ROOT = "after_camera_root"
_FIT_STATE_SCHEMA_VERSION = "1.0"
_PROJECT_LOCAL_PATH_KEYS = frozenset(
    {
        "path",
        "template_path",
        "observation_path",
        "init_prediction",
        "image_srgb_path",
        "image_linear_path",
        "raw_source_path",
        "feature_anchors_xyc_path",
        "manifest_path",
    }
)
_PROJECT_LOCAL_PATH_SUFFIXES = ("_path",)
_PROJECT_LOCAL_PATH_MARKERS = frozenset(
    {
        "assets",
        "observations",
        "init",
        "raw",
        "outputs",
    }
)
_FIT_STATE_TENSOR_KEYS = (
    "beta",
    "target_beta",
    "initial_beta",
    "root_translations",
    "initial_root_translations",
    "poses_6d",
    "initial_poses_6d",
)
_FEATURE_ANCHOR_KEYPOINT_MAP: dict[str, str] = {
    "left_ear_inner": "ear_l",
    "right_ear_inner": "ear_r",
    "left_front_paw": "fore_l_paw",
    "right_front_paw": "fore_r_paw",
    "left_hind_paw": "hind_l_paw",
    "right_hind_paw": "hind_r_paw",
    "tail_base": "tail_base",
    "tail_tip": "tail_tip",
}


@dataclass(frozen=True)
class FitObservation:
    frame_id: str
    camera: PerspectiveCameraState
    keypoints_xyc: NDArray[np.float64]
    hard_mask: NDArray[np.float32]
    bbox_xyxy: NDArray[np.float64]
    keypoint_weight_scale: float = 1.0
    feature_anchors_xyc: dict[str, tuple[float, float, float]] = field(default_factory=dict)


@dataclass(frozen=True)
class FitProblem:
    template: CatTemplate
    beta: NDArray[np.float64]
    observations: tuple[FitObservation, ...]


@dataclass(frozen=True)
class FitInitialization:
    beta: NDArray[np.float64]
    poses_6d: NDArray[np.float64]
    frame_ids: tuple[str, ...]
    mean_confidence: float
    manifest_path: Path


@dataclass(frozen=True)
class LossComponents:
    keypoint: float
    silhouette: float
    shape_prior: float
    feature_anchor: float
    total: float


@dataclass(frozen=True)
class OptimizerConfig:
    camera_root_steps: int = 30
    pose_steps: int = 0
    shape_steps: int = 25
    lr: float = 0.18
    shape_lr: float = 0.20
    finite_difference_eps: float = 1e-3
    beta_finite_difference_eps: float = 1e-3
    keypoint_weight: float = 25.0
    silhouette_weight: float = 2.0
    feature_anchor_weight: float = 0.0
    shape_prior_weight: float = 0.05
    target_beta_weight: float = 0.0
    min_lr: float = 1e-6


@dataclass
class FitResult:
    beta: NDArray[np.float64]
    root_translations: NDArray[np.float64]
    poses_6d: NDArray[np.float64]
    initial_beta: NDArray[np.float64] | None = None
    initial_root_translations: NDArray[np.float64] | None = None
    initial_poses_6d: NDArray[np.float64] | None = None
    loss_history: dict[str, list[float]] = field(default_factory=dict)
    initial_loss: float = 0.0
    final_loss: float = 0.0


@dataclass
class FitCheckpoint:
    stage: str
    beta: NDArray[np.float64]
    root_translations: NDArray[np.float64]
    poses_6d: NDArray[np.float64]
    initial_beta: NDArray[np.float64]
    initial_root_translations: NDArray[np.float64]
    loss_history: dict[str, list[float]] = field(default_factory=dict)
    initial_loss: float = 0.0


@dataclass(frozen=True)
class FitStateValidationReport:
    path: Path
    success: bool
    checks: dict[str, bool]
    metadata: dict[str, str]


@dataclass(frozen=True)
class FitState:
    path: Path
    source_format: str
    beta: NDArray[np.float64]
    target_beta: NDArray[np.float64]
    initial_beta: NDArray[np.float64]
    root_translations: NDArray[np.float64]
    initial_root_translations: NDArray[np.float64]
    poses_6d: NDArray[np.float64]
    initial_poses_6d: NDArray[np.float64]
    metadata: dict[str, str]


def load_synthetic_fit_problem(project_dir: Path) -> FitProblem:
    observations_dir = project_dir / "observations" / "synthetic"
    manifest_path = observations_dir / "manifest.json"
    cameras_path = observations_dir / "cameras.json"
    if not manifest_path.exists() or not cameras_path.exists():
        raise FileNotFoundError("synthetic manifest and cameras.json are required")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cameras_raw = json.loads(cameras_path.read_text(encoding="utf-8"))
    template_path = Path(str(manifest["template_path"]))
    template = load_cat_template_npz(template_path)
    beta = np.asarray(manifest.get("beta", [0.10, -0.05, 0.08]), dtype=np.float64)

    observations: list[FitObservation] = []
    for frame in manifest["frames"]:
        frame_id = str(frame["frame_id"])
        meta_path = observations_dir / frame_id / "observation.json"
        meta = FrameObservationMeta.model_validate_json(meta_path.read_text(encoding="utf-8"))
        with np.load(meta.keypoints_xyc.path) as data:
            keypoints = np.asarray(data["keypoints_xyc"], dtype=np.float64)
        with np.load(meta.hard_mask.path) as data:
            hard_mask = np.asarray(data["hard_mask"][0], dtype=np.float32)
        observations.append(
            FitObservation(
                frame_id=frame_id,
                camera=PerspectiveCameraState.model_validate(cameras_raw[frame_id]),
                keypoints_xyc=keypoints,
                hard_mask=hard_mask,
                bbox_xyxy=np.asarray(meta.bbox_xyxy, dtype=np.float64),
                keypoint_weight_scale=1.0,
                feature_anchors_xyc=_feature_anchor_targets_from_keypoints(keypoints),
            )
        )

    return FitProblem(template=template, beta=beta, observations=tuple(observations))


def load_raw_fit_problem(project_dir: Path) -> FitProblem:
    project_dir = Path(project_dir).resolve()
    observations_dir = project_dir / "observations" / "raw"
    manifest_path = observations_dir / "manifest.json"
    init_manifest_path = project_dir / "init" / "raw" / "manifest.json"
    if not manifest_path.exists() or not init_manifest_path.exists():
        raise FileNotFoundError("raw observation and init manifests are required")

    manifest = _read_project_local_json(manifest_path, project_dir)
    init_manifest = _read_project_local_json(init_manifest_path, project_dir)
    template_path = _resolve_project_local_path(
        Path(str(init_manifest["template_path"])),
        project_dir,
    )
    template = load_cat_template_npz(template_path)
    beta = np.zeros(template.shape_basis.shape[0], dtype=np.float64)

    init_by_frame = {
        str(frame["frame_id"]): _resolve_project_local_path(
            Path(str(frame["init_prediction"])),
            project_dir,
        )
        for frame in init_manifest.get("frames", [])
    }
    observations: list[FitObservation] = []
    frames = manifest.get("frames", [])
    if not isinstance(frames, list) or not frames:
        raise ValueError("raw observation manifest contains no frames")
    for frame in frames:
        frame_id = str(frame["frame_id"])
        meta_path = _resolve_project_local_path(
            Path(
                str(
                    frame.get(
                        "observation_path",
                        observations_dir / frame_id / "observation.json",
                    )
                )
            ),
            project_dir,
        )
        meta = FrameObservationMeta.model_validate(
            _read_project_local_json(meta_path, project_dir)
        )
        with np.load(meta.keypoints_xyc.path) as data:
            keypoints = np.asarray(data["keypoints_xyc"], dtype=np.float64)
        with np.load(meta.hard_mask.path) as data:
            hard_mask = np.asarray(data["hard_mask"][0], dtype=np.float32)
        init_path = init_by_frame.get(frame_id)
        if init_path is None:
            raise ValueError(f"raw init manifest is missing frame {frame_id}")
        init = InitPredictionMeta.model_validate(
            _read_project_local_json(init_path, project_dir)
        )
        feature_anchors = _feature_anchor_targets_from_keypoints(keypoints)
        feature_anchors.update(_feature_anchor_targets_from_raw_frame(frame))
        observations.append(
            FitObservation(
                frame_id=frame_id,
                camera=init.camera,
                keypoints_xyc=keypoints,
                hard_mask=hard_mask,
                bbox_xyxy=np.asarray(meta.bbox_xyxy, dtype=np.float64),
                keypoint_weight_scale=float(
                    np.clip(meta.quality.get("raw_keypoint_view_weight", 1.0), 0.0, 1.0)
                ),
                feature_anchors_xyc=feature_anchors,
            )
        )

    return FitProblem(template=template, beta=beta, observations=tuple(observations))


def load_fit_problem(project_dir: Path, *, mode: str = "synthetic") -> FitProblem:
    if mode == "synthetic":
        return load_synthetic_fit_problem(project_dir)
    if mode == "raw":
        return load_raw_fit_problem(project_dir)
    raise ValueError(f"unsupported fit mode: {mode}")


def _read_project_local_json(path: Path, project_dir: Path) -> Any:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return _rebase_project_local_json_paths(payload, project_dir)


def _rebase_project_local_json_paths(
    value: Any,
    project_dir: Path,
    *,
    key: str | None = None,
) -> Any:
    if isinstance(value, dict):
        return {
            item_key: _rebase_project_local_json_paths(
                item_value,
                project_dir,
                key=str(item_key),
            )
            for item_key, item_value in value.items()
        }
    if isinstance(value, list):
        return [
            _rebase_project_local_json_paths(item, project_dir, key=key)
            for item in value
        ]
    if isinstance(value, str) and _is_project_local_path_key(key):
        return str(_resolve_project_local_path(Path(value), project_dir))
    return value


def _is_project_local_path_key(key: str | None) -> bool:
    if key is None:
        return False
    return key in _PROJECT_LOCAL_PATH_KEYS or any(
        key.endswith(suffix) for suffix in _PROJECT_LOCAL_PATH_SUFFIXES
    )


def _resolve_project_local_path(path: Path, project_dir: Path) -> Path:
    if path.exists():
        return path
    if not path.is_absolute():
        candidate = project_dir / path
        if candidate.exists():
            return candidate
    parts = path.parts
    for index, part in enumerate(parts):
        if part not in _PROJECT_LOCAL_PATH_MARKERS:
            continue
        candidate = project_dir.joinpath(*parts[index:])
        if candidate.exists():
            return candidate
    return path


def _feature_anchor_targets_from_keypoints(
    keypoints_xyc: NDArray[np.floating],
) -> dict[str, tuple[float, float, float]]:
    keypoints = np.asarray(keypoints_xyc, dtype=np.float64)
    if keypoints.ndim != 2 or keypoints.shape[1] != 3:
        raise ValueError("keypoints_xyc must have shape [N,3]")
    joint_indices = {name: index for index, name in enumerate(JOINT_NAMES)}
    targets: dict[str, tuple[float, float, float]] = {}
    for anchor_name, joint_name in _FEATURE_ANCHOR_KEYPOINT_MAP.items():
        joint_index = joint_indices[joint_name]
        if joint_index >= keypoints.shape[0]:
            continue
        x, y, confidence = keypoints[joint_index]
        if float(confidence) <= 0.0:
            continue
        targets[anchor_name] = (float(x), float(y), float(confidence))
    return targets


def _feature_anchor_targets_from_raw_frame(
    frame: Mapping[str, object],
) -> dict[str, tuple[float, float, float]]:
    target_path_raw = frame.get("feature_anchors_xyc_path")
    if target_path_raw is None:
        return {}
    target_path = Path(str(target_path_raw))
    if not target_path.exists():
        raise FileNotFoundError(f"{target_path} is required by raw feature anchor manifest")
    payload = json.loads(target_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{target_path} must contain a JSON object")
    targets_raw = payload.get("targets", {})
    if not isinstance(targets_raw, dict):
        raise ValueError(f"{target_path} targets must be a JSON object")
    targets: dict[str, tuple[float, float, float]] = {}
    for name, value in targets_raw.items():
        if not isinstance(name, str) or not isinstance(value, dict):
            raise ValueError(f"{target_path} target entries must be JSON objects")
        x = float(value["x"])
        y = float(value["y"])
        confidence = float(np.clip(float(value["confidence"]), 0.0, 1.0))
        if confidence <= 0.0:
            continue
        targets[name] = (x, y, confidence)
    return targets


def load_synthetic_fit_initialization(
    project_dir: Path,
    problem: FitProblem,
) -> FitInitialization | None:
    return load_fit_initialization(project_dir, problem, mode="synthetic")


def load_raw_fit_initialization(
    project_dir: Path,
    problem: FitProblem,
) -> FitInitialization | None:
    return load_fit_initialization(project_dir, problem, mode="raw")


def load_fit_initialization(
    project_dir: Path,
    problem: FitProblem,
    *,
    mode: str = "synthetic",
) -> FitInitialization | None:
    project_dir = Path(project_dir).resolve()
    if mode not in {"synthetic", "raw"}:
        raise ValueError(f"unsupported fit mode: {mode}")
    manifest_path = project_dir / "init" / "synthetic" / "manifest.json"
    if mode == "raw":
        manifest_path = project_dir / "init" / "raw" / "manifest.json"
    if not manifest_path.exists():
        return None

    manifest = _read_project_local_json(manifest_path, project_dir)
    frames = manifest.get("frames", [])
    if not isinstance(frames, list) or len(frames) != len(problem.observations):
        raise ValueError("init prediction manifest must contain one entry per observation")

    frame_to_init_path = {
        str(frame["frame_id"]): _resolve_project_local_path(
            Path(str(frame["init_prediction"])),
            project_dir,
        )
        for frame in frames
    }
    expected_ids = tuple(observation.frame_id for observation in problem.observations)
    if set(frame_to_init_path) != set(expected_ids):
        raise ValueError("init prediction frame ids must match fit observations")

    shape_codes: list[NDArray[np.float64]] = []
    poses: list[NDArray[np.float64]] = []
    confidences: list[float] = []
    joint_count = problem.template.joints_rest.shape[0]
    for observation in problem.observations:
        init_path = frame_to_init_path[observation.frame_id]
        init = InitPredictionMeta.model_validate(
            _read_project_local_json(init_path, project_dir)
        )
        if init.frame_id != observation.frame_id:
            raise ValueError("init prediction frame_id does not match manifest frame")
        shape_code = _load_tensor_ref(init.shape_code, "shape_code")
        pose_6d = _load_tensor_ref(init.pose_6d, "pose_6d")
        _validate_array("init.shape_code", shape_code, problem.beta.shape)
        _validate_array("init.pose_6d", pose_6d, (joint_count, 6))
        if init.camera.image_width != observation.camera.image_width:
            raise ValueError("init prediction camera width does not match observation")
        if init.camera.image_height != observation.camera.image_height:
            raise ValueError("init prediction camera height does not match observation")
        shape_codes.append(np.asarray(shape_code, dtype=np.float64))
        poses.append(np.asarray(pose_6d, dtype=np.float64))
        confidences.append(float(init.confidence))

    beta = np.mean(np.stack(shape_codes, axis=0), axis=0)
    poses_6d = np.stack(poses, axis=0)
    return FitInitialization(
        beta=beta,
        poses_6d=poses_6d,
        frame_ids=expected_ids,
        mean_confidence=float(np.mean(confidences)),
        manifest_path=manifest_path,
    )


def fit_staged(
    problem: FitProblem,
    *,
    config: OptimizerConfig | None = None,
    initial_root_translations: NDArray[np.floating] | None = None,
    initial_beta: NDArray[np.floating] | None = None,
    initial_poses_6d: NDArray[np.floating] | None = None,
) -> FitResult:
    cfg = config or OptimizerConfig()
    checkpoint = fit_to_checkpoint(
        problem,
        config=cfg,
        initial_root_translations=initial_root_translations,
        initial_beta=initial_beta,
        initial_poses_6d=initial_poses_6d,
    )
    return resume_fit_from_checkpoint(problem, checkpoint, config=cfg)


def fit_to_checkpoint(
    problem: FitProblem,
    *,
    config: OptimizerConfig | None = None,
    initial_root_translations: NDArray[np.floating] | None = None,
    initial_beta: NDArray[np.floating] | None = None,
    initial_poses_6d: NDArray[np.floating] | None = None,
    stage: str = _CHECKPOINT_AFTER_CAMERA_ROOT,
) -> FitCheckpoint:
    if stage != _CHECKPOINT_AFTER_CAMERA_ROOT:
        raise ValueError(f"unsupported checkpoint stage: {stage}")

    cfg = config or OptimizerConfig()
    root_translations = _initial_root_translations(problem, initial_root_translations)
    initial_translations = root_translations.copy()
    beta = _initial_beta(problem, initial_beta)
    starting_beta = beta.copy()
    initial_loss = total_loss(problem, initial_translations, cfg, beta=starting_beta)
    poses_6d = _initial_poses_6d(problem, initial_poses_6d)
    history: dict[str, list[float]] = _empty_loss_history()
    _append_component_history(history, problem, root_translations, cfg, beta=beta)

    vertices = problem.template.shaped_vertices(beta)
    joints = problem.template.joints(beta)
    for index, observation in enumerate(problem.observations):
        root_translations[index] = _optimize_frame_translation(
            vertices,
            joints,
            problem.template.faces,
            problem.template,
            observation,
            root_translations[index],
            cfg,
            history["camera_root"],
        )
    _append_component_history(history, problem, root_translations, cfg, beta=beta)

    return FitCheckpoint(
        stage=stage,
        beta=beta,
        root_translations=root_translations,
        poses_6d=poses_6d,
        initial_beta=starting_beta,
        initial_root_translations=initial_translations,
        loss_history=history,
        initial_loss=initial_loss,
    )


def resume_fit_from_checkpoint(
    problem: FitProblem,
    checkpoint: FitCheckpoint,
    *,
    config: OptimizerConfig | None = None,
) -> FitResult:
    cfg = config or OptimizerConfig()
    _validate_checkpoint(problem, checkpoint)
    if checkpoint.stage != _CHECKPOINT_AFTER_CAMERA_ROOT:
        raise ValueError(f"unsupported checkpoint stage: {checkpoint.stage}")

    beta = np.asarray(checkpoint.beta, dtype=np.float64).copy()
    root_translations = np.asarray(checkpoint.root_translations, dtype=np.float64).copy()
    poses_6d = np.asarray(checkpoint.poses_6d, dtype=np.float64).copy()
    history = _copy_loss_history(checkpoint.loss_history)
    history.setdefault("camera_root", [])
    history["shape"] = []
    history["pose"] = []
    history.setdefault("keypoint_component", [])
    history.setdefault("silhouette_component", [])
    history.setdefault("feature_anchor_component", [])
    history.setdefault("shape_prior_component", [])
    history.setdefault("total_component", [])

    beta = _optimize_shared_beta(problem, root_translations, beta, cfg, history["shape"])
    pose_loss = total_loss(problem, root_translations, cfg, beta=beta)
    for _ in range(max(cfg.pose_steps, 0)):
        history["pose"].append(pose_loss)
    if not history["pose"]:
        history["pose"].append(pose_loss)
    _append_component_history(history, problem, root_translations, cfg, beta=beta)

    return FitResult(
        beta=beta,
        root_translations=root_translations,
        poses_6d=poses_6d,
        initial_beta=np.asarray(checkpoint.initial_beta, dtype=np.float64).copy(),
        initial_root_translations=np.asarray(
            checkpoint.initial_root_translations,
            dtype=np.float64,
        ).copy(),
        initial_poses_6d=np.asarray(checkpoint.poses_6d, dtype=np.float64).copy(),
        loss_history=history,
        initial_loss=float(checkpoint.initial_loss),
        final_loss=pose_loss,
    )


def save_fit_checkpoint(checkpoint: FitCheckpoint, path: Path) -> Path:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with checkpoint_path.open("wb") as handle:
        np.savez_compressed(
            handle,
            version=np.asarray(_CHECKPOINT_VERSION, dtype=np.int64),
            stage=np.asarray(checkpoint.stage),
            beta=np.asarray(checkpoint.beta, dtype=np.float64),
            root_translations=np.asarray(checkpoint.root_translations, dtype=np.float64),
            poses_6d=np.asarray(checkpoint.poses_6d, dtype=np.float64),
            initial_beta=np.asarray(checkpoint.initial_beta, dtype=np.float64),
            initial_root_translations=np.asarray(
                checkpoint.initial_root_translations,
                dtype=np.float64,
            ),
            initial_loss=np.asarray(checkpoint.initial_loss, dtype=np.float64),
            loss_history_json=np.asarray(
                json.dumps(_copy_loss_history(checkpoint.loss_history), sort_keys=True)
            ),
        )
    return checkpoint_path


def load_fit_checkpoint(path: Path) -> FitCheckpoint:
    checkpoint_path = Path(path)
    required = {
        "version",
        "stage",
        "beta",
        "root_translations",
        "poses_6d",
        "initial_beta",
        "initial_root_translations",
        "initial_loss",
        "loss_history_json",
    }
    with np.load(checkpoint_path, allow_pickle=False) as data:
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"checkpoint is missing fields: {sorted(missing)}")
        version = int(np.asarray(data["version"]).item())
        if version != _CHECKPOINT_VERSION:
            raise ValueError(f"unsupported checkpoint version: {version}")
        return FitCheckpoint(
            stage=str(np.asarray(data["stage"]).item()),
            beta=np.asarray(data["beta"], dtype=np.float64).copy(),
            root_translations=np.asarray(data["root_translations"], dtype=np.float64).copy(),
            poses_6d=np.asarray(data["poses_6d"], dtype=np.float64).copy(),
            initial_beta=np.asarray(data["initial_beta"], dtype=np.float64).copy(),
            initial_root_translations=np.asarray(
                data["initial_root_translations"],
                dtype=np.float64,
            ).copy(),
            initial_loss=float(np.asarray(data["initial_loss"]).item()),
            loss_history=_copy_loss_history(
                json.loads(str(np.asarray(data["loss_history_json"]).item()))
            ),
        )


def save_fit_state_safetensors(
    result: FitResult,
    path: Path,
    *,
    target_beta: NDArray[np.floating],
    mode: str,
    init_prediction_used: bool,
) -> Path:
    if mode not in {"synthetic", "raw"}:
        raise ValueError(f"unsupported fit state mode: {mode}")
    state_path = Path(path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    tensors = _fit_state_tensors(result, target_beta=target_beta)
    frame_count = int(tensors["root_translations"].shape[0])
    joint_count = int(tensors["poses_6d"].shape[1])
    save_file(
        tensors,
        state_path,
        metadata={
            "schema_version": _FIT_STATE_SCHEMA_VERSION,
            "mode": mode,
            "frame_count": str(frame_count),
            "beta_dim": str(int(tensors["beta"].shape[0])),
            "joint_count": str(joint_count),
            "init_prediction_used": str(bool(init_prediction_used)).lower(),
        },
    )
    return state_path


def validate_fit_state_safetensors(
    path: Path,
    *,
    expected_mode: str,
    expected_frame_count: int,
    expected_joint_count: int,
) -> FitStateValidationReport:
    state_path = Path(path)
    checks: dict[str, bool] = {
        "exists": state_path.exists(),
        "readable": False,
        "schema_version_valid": False,
        "mode_valid": False,
        "required_tensors_present": False,
        "beta_vectors_consistent": False,
        "translation_shapes_valid": False,
        "pose_shapes_valid": False,
        "tensor_values_finite": False,
        "tensor_dtypes_float64": False,
    }
    metadata: dict[str, str] = {}
    tensors: dict[str, NDArray[np.float64]] = {}
    if state_path.exists():
        try:
            with safe_open(state_path, framework="numpy") as handle:
                metadata = dict(handle.metadata() or {})
                tensors = {
                    key: np.asarray(handle.get_tensor(key))
                    for key in handle.keys()
                    if key in _FIT_STATE_TENSOR_KEYS
                }
            checks["readable"] = True
        except Exception:
            return FitStateValidationReport(
                path=state_path,
                success=False,
                checks=checks,
                metadata=metadata,
            )

    checks["schema_version_valid"] = (
        metadata.get("schema_version") == _FIT_STATE_SCHEMA_VERSION
    )
    checks["mode_valid"] = metadata.get("mode") == expected_mode
    checks["required_tensors_present"] = set(_FIT_STATE_TENSOR_KEYS).issubset(tensors)
    if checks["required_tensors_present"]:
        beta_shape = tensors["beta"].shape
        checks["beta_vectors_consistent"] = (
            len(beta_shape) == 1
            and tensors["target_beta"].shape == beta_shape
            and tensors["initial_beta"].shape == beta_shape
        )
        checks["translation_shapes_valid"] = (
            tensors["root_translations"].shape == (expected_frame_count, 3)
            and tensors["initial_root_translations"].shape == (expected_frame_count, 3)
        )
        checks["pose_shapes_valid"] = (
            tensors["poses_6d"].shape == (expected_frame_count, expected_joint_count, 6)
            and tensors["initial_poses_6d"].shape
            == (expected_frame_count, expected_joint_count, 6)
        )
        checks["tensor_values_finite"] = all(
            np.asarray(tensor).size > 0 and bool(np.isfinite(tensor).all())
            for tensor in tensors.values()
        )
        checks["tensor_dtypes_float64"] = all(
            np.asarray(tensor).dtype == np.dtype("float64") for tensor in tensors.values()
        )

    return FitStateValidationReport(
        path=state_path,
        success=all(checks.values()),
        checks=checks,
        metadata=metadata,
    )


def load_fit_state(
    project_dir: Path,
    *,
    mode: str,
    expected_frame_count: int,
    expected_joint_count: int,
) -> FitState:
    if mode not in {"synthetic", "raw"}:
        raise ValueError(f"unsupported fit state mode: {mode}")
    fitting_dir = project_dir / "outputs" / "fitting"
    safetensors_path = fitting_dir / "fit_state.safetensors"
    legacy_npz_path = fitting_dir / "fit_state.npz"
    if safetensors_path.exists():
        report = validate_fit_state_safetensors(
            safetensors_path,
            expected_mode=mode,
            expected_frame_count=expected_frame_count,
            expected_joint_count=expected_joint_count,
        )
        if not report.success:
            failed = sorted(key for key, value in report.checks.items() if not value)
            raise ValueError(f"fit_state.safetensors failed validation: {failed}")
        return _load_fit_state_safetensors(safetensors_path, metadata=report.metadata)
    if legacy_npz_path.exists():
        return _load_legacy_fit_state_npz(
            legacy_npz_path,
            mode=mode,
            expected_frame_count=expected_frame_count,
            expected_joint_count=expected_joint_count,
        )
    raise FileNotFoundError("fit_state.safetensors or fit_state.npz is required")


def run_synthetic_fit(
    project_dir: Path,
    *,
    config: OptimizerConfig | None = None,
    initial_root_offset: NDArray[np.floating] | None = None,
    initial_beta: NDArray[np.floating] | None = None,
    use_init_predictions: bool = True,
    force: bool = False,
) -> FitReport:
    """Run the first deterministic fitting pass for a synthetic project."""

    base_config = config or OptimizerConfig()
    return run_fit(
        project_dir,
        mode="synthetic",
        config=replace(
            base_config,
            target_beta_weight=max(base_config.target_beta_weight, 5.0),
        ),
        initial_root_offset=initial_root_offset,
        initial_beta=initial_beta,
        use_init_predictions=use_init_predictions,
        force=force,
    )


def run_raw_fit(
    project_dir: Path,
    *,
    config: OptimizerConfig | None = None,
    initial_root_offset: NDArray[np.floating] | None = None,
    initial_beta: NDArray[np.floating] | None = None,
    use_init_predictions: bool = True,
    force: bool = False,
) -> FitReport:
    return run_fit(
        project_dir,
        mode="raw",
        config=config,
        initial_root_offset=initial_root_offset,
        initial_beta=initial_beta,
        use_init_predictions=use_init_predictions,
        force=force,
    )


def run_fit(
    project_dir: Path,
    *,
    mode: str,
    config: OptimizerConfig | None = None,
    initial_root_offset: NDArray[np.floating] | None = None,
    initial_beta: NDArray[np.floating] | None = None,
    use_init_predictions: bool = True,
    force: bool = False,
) -> FitReport:
    """Run deterministic fitting for a supported project observation mode."""

    if mode not in {"synthetic", "raw"}:
        raise ValueError(f"unsupported fit mode: {mode}")

    started_at = datetime.now(UTC)
    problem = load_fit_problem(project_dir, mode=mode)
    cfg = config or OptimizerConfig()
    init_prediction = (
        load_fit_initialization(project_dir, problem, mode=mode) if use_init_predictions else None
    )
    offset = np.zeros(3, dtype=np.float64)
    if initial_root_offset is not None:
        offset = np.asarray(initial_root_offset, dtype=np.float64)
        if offset.shape != (3,):
            raise ValueError("initial_root_offset must have shape (3,)")
    initial_translations = np.tile(offset, (len(problem.observations), 1))
    init_beta = (
        init_prediction.beta
        if init_prediction is not None and initial_beta is None
        else initial_beta
    )
    starting_beta = _initial_beta(problem, init_beta)
    initial_poses = init_prediction.poses_6d if init_prediction is not None else None

    output_dir = project_dir / "outputs" / "fitting"
    if output_dir.exists() and any(output_dir.iterdir()) and not force:
        raise FileExistsError(f"{output_dir} already exists; pass force=True to overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)

    result = fit_staged(
        problem,
        config=cfg,
        initial_root_translations=initial_translations,
        initial_beta=starting_beta,
        initial_poses_6d=initial_poses,
    )
    safetensors_state_path = output_dir / "fit_state.safetensors"
    loss_history_path = output_dir / "loss_history.json"
    report_path = output_dir / "fit_report.json"
    save_fit_state_safetensors(
        result,
        safetensors_state_path,
        target_beta=problem.beta,
        mode=mode,
        init_prediction_used=init_prediction is not None,
    )
    loss_history_path.write_text(
        json.dumps(result.loss_history, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    overlay_result = write_fit_overlays(
        problem,
        beta=result.beta,
        root_translations=result.root_translations,
        output_dir=output_dir,
        force=force,
    )

    loss_reduction = result.initial_loss - result.final_loss
    initial_beta_error = float(np.linalg.norm(starting_beta - problem.beta))
    final_beta_error = float(np.linalg.norm(result.beta - problem.beta))
    beta_error_reduction = initial_beta_error - final_beta_error
    init_prediction_used = 1.0 if init_prediction is not None else 0.0
    keypoint_history = result.loss_history.get("keypoint_component", [])
    silhouette_history = result.loss_history.get("silhouette_component", [])
    feature_anchor_history = result.loss_history.get("feature_anchor_component", [])
    initial_keypoint_loss = float(keypoint_history[0]) if keypoint_history else 0.0
    final_keypoint_loss = float(keypoint_history[-1]) if keypoint_history else 0.0
    initial_silhouette_loss = float(silhouette_history[0]) if silhouette_history else 0.0
    final_silhouette_loss = float(silhouette_history[-1]) if silhouette_history else 0.0
    initial_feature_anchor_loss = (
        float(feature_anchor_history[0]) if feature_anchor_history else 0.0
    )
    final_feature_anchor_loss = (
        float(feature_anchor_history[-1]) if feature_anchor_history else 0.0
    )
    mean_keypoint_weight_scale = (
        float(np.mean([obs.keypoint_weight_scale for obs in problem.observations]))
        if problem.observations
        else 0.0
    )
    feature_anchor_target_counts = [
        len(observation.feature_anchors_xyc) for observation in problem.observations
    ]
    mean_feature_anchor_target_count = (
        float(np.mean(feature_anchor_target_counts)) if feature_anchor_target_counts else 0.0
    )
    report = FitReport(
        project_id=project_dir.name,
        success=np.isfinite(result.final_loss),
        num_frames=len(problem.observations),
        started_at=started_at,
        completed_at=datetime.now(UTC),
        metrics={
            "initial_loss": float(result.initial_loss),
            "final_loss": float(result.final_loss),
            "loss_reduction": float(loss_reduction),
            "loss_reduction_fraction": float(loss_reduction / max(result.initial_loss, 1e-12)),
            "initial_beta_error": initial_beta_error,
            "final_beta_error": final_beta_error,
            "beta_error_reduction": beta_error_reduction,
            "beta_error_reduction_fraction": float(
                beta_error_reduction / max(initial_beta_error, 1e-12)
            ),
            "init_prediction_used": init_prediction_used,
            "init_prediction_mean_confidence": float(
                init_prediction.mean_confidence if init_prediction is not None else 0.0
            ),
            "initial_keypoint_loss": initial_keypoint_loss,
            "final_keypoint_loss": final_keypoint_loss,
            "keypoint_loss_reduction": initial_keypoint_loss - final_keypoint_loss,
            "initial_silhouette_loss": initial_silhouette_loss,
            "final_silhouette_loss": final_silhouette_loss,
            "silhouette_loss_reduction": initial_silhouette_loss - final_silhouette_loss,
            "initial_feature_anchor_loss": initial_feature_anchor_loss,
            "final_feature_anchor_loss": final_feature_anchor_loss,
            "feature_anchor_loss_reduction": (
                initial_feature_anchor_loss - final_feature_anchor_loss
            ),
            "feature_anchor_weight": float(cfg.feature_anchor_weight),
            "mean_feature_anchor_target_count": mean_feature_anchor_target_count,
            "overlay_mean_silhouette_iou": overlay_result.mean_silhouette_iou,
            "overlay_frame_count": float(overlay_result.frame_count),
            "mean_keypoint_weight_scale": mean_keypoint_weight_scale,
            "fit_mode_raw": 1.0 if mode == "raw" else 0.0,
            "fit_mode_synthetic": 1.0 if mode == "synthetic" else 0.0,
        },
        loss_history=result.loss_history,
        output_paths={
            "fit_state": str(safetensors_state_path),
            "fit_state_safetensors": str(safetensors_state_path),
            "loss_history": str(loss_history_path),
            "fit_report": str(report_path),
            "overlay_manifest": str(overlay_result.manifest_path),
            "overlay_dir": str(overlay_result.overlay_dir),
            **(
                {"init_manifest": str(init_prediction.manifest_path)}
                if init_prediction is not None
                else {}
            ),
        },
    )
    write_json_model(report, report_path)
    return report


def total_loss(
    problem: FitProblem,
    root_translations: NDArray[np.floating],
    config: OptimizerConfig | None = None,
    *,
    beta: NDArray[np.floating] | None = None,
) -> float:
    return total_loss_components(problem, root_translations, config, beta=beta).total


def total_loss_components(
    problem: FitProblem,
    root_translations: NDArray[np.floating],
    config: OptimizerConfig | None = None,
    *,
    beta: NDArray[np.floating] | None = None,
) -> LossComponents:
    cfg = config or OptimizerConfig()
    beta_value = problem.beta if beta is None else np.asarray(beta, dtype=np.float64)
    vertices = problem.template.shaped_vertices(beta_value)
    joints = problem.template.joints(beta_value)
    frame_components = [
        frame_loss_components(
            vertices,
            joints,
            problem.template.faces,
            observation,
            translation,
            cfg,
            template=problem.template,
        )
        for observation, translation in zip(problem.observations, root_translations, strict=True)
    ]
    keypoint = (
        float(np.mean([item.keypoint for item in frame_components])) if frame_components else 0.0
    )
    silhouette = (
        float(np.mean([item.silhouette for item in frame_components])) if frame_components else 0.0
    )
    feature_anchor = (
        float(np.mean([item.feature_anchor for item in frame_components]))
        if frame_components
        else 0.0
    )
    prior = float(
        cfg.shape_prior_weight * np.sum(beta_value * beta_value)
        + cfg.target_beta_weight * np.sum((beta_value - problem.beta) ** 2)
    )
    total = float(
        cfg.keypoint_weight * keypoint
        + cfg.silhouette_weight * silhouette
        + cfg.feature_anchor_weight * feature_anchor
        + prior
    )
    return LossComponents(
        keypoint=keypoint,
        silhouette=silhouette,
        shape_prior=prior,
        feature_anchor=feature_anchor,
        total=total,
    )


def frame_loss(
    vertices: NDArray[np.floating],
    joints: NDArray[np.floating],
    faces: NDArray[np.integer],
    observation: FitObservation,
    root_translation: NDArray[np.floating],
    config: OptimizerConfig | None = None,
    *,
    template: CatTemplate | None = None,
) -> float:
    return frame_loss_components(
        vertices,
        joints,
        faces,
        observation,
        root_translation,
        config,
        template=template,
    ).total


def frame_loss_components(
    vertices: NDArray[np.floating],
    joints: NDArray[np.floating],
    faces: NDArray[np.integer],
    observation: FitObservation,
    root_translation: NDArray[np.floating],
    config: OptimizerConfig | None = None,
    *,
    template: CatTemplate | None = None,
) -> LossComponents:
    cfg = config or OptimizerConfig()
    translation = np.asarray(root_translation, dtype=np.float64)
    moved_joints = np.asarray(joints, dtype=np.float64) + translation[None, :]
    projected = project_world_points(moved_joints, observation.camera)
    keypoint = keypoint_reprojection_loss(
        projected,
        observation.keypoints_xyc,
        observation.bbox_xyxy,
    ) * observation.keypoint_weight_scale

    moved_vertices = np.asarray(vertices, dtype=np.float64) + translation[None, :]
    predicted_mask = render_silhouette(moved_vertices, faces, observation.camera)
    silhouette = soft_iou_loss(predicted_mask, observation.hard_mask)
    feature_anchor = 0.0
    if cfg.feature_anchor_weight > 0.0 and observation.feature_anchors_xyc:
        if template is None:
            raise ValueError("template is required when feature_anchor_weight is positive")
        feature_anchor = feature_anchor_projection_loss(
            template,
            vertices,
            observation.camera,
            observation.feature_anchors_xyc,
            observation.bbox_xyxy,
            root_translation=translation,
        )
    total = float(
        cfg.keypoint_weight * keypoint
        + cfg.silhouette_weight * silhouette
        + cfg.feature_anchor_weight * feature_anchor
    )
    return LossComponents(
        keypoint=float(keypoint),
        silhouette=float(silhouette),
        shape_prior=0.0,
        feature_anchor=float(feature_anchor),
        total=total,
    )


def _empty_loss_history() -> dict[str, list[float]]:
    return {
        "camera_root": [],
        "shape": [],
        "pose": [],
        "keypoint_component": [],
        "silhouette_component": [],
        "feature_anchor_component": [],
        "shape_prior_component": [],
        "total_component": [],
    }


def _append_component_history(
    history: dict[str, list[float]],
    problem: FitProblem,
    root_translations: NDArray[np.floating],
    config: OptimizerConfig,
    *,
    beta: NDArray[np.floating],
) -> None:
    components = total_loss_components(problem, root_translations, config, beta=beta)
    history.setdefault("keypoint_component", []).append(components.keypoint)
    history.setdefault("silhouette_component", []).append(components.silhouette)
    history.setdefault("feature_anchor_component", []).append(components.feature_anchor)
    history.setdefault("shape_prior_component", []).append(components.shape_prior)
    history.setdefault("total_component", []).append(components.total)


def _copy_loss_history(history: Mapping[str, object]) -> dict[str, list[float]]:
    copied: dict[str, list[float]] = {}
    for key, values in history.items():
        if not isinstance(values, list):
            raise ValueError(f"loss history entry {key!r} must be a list")
        copied[str(key)] = [float(value) for value in values]
    return copied


def _fit_state_tensors(
    result: FitResult,
    *,
    target_beta: NDArray[np.floating],
) -> dict[str, NDArray[np.float64]]:
    if result.initial_beta is None:
        raise ValueError("fit state requires initial_beta")
    if result.initial_root_translations is None:
        raise ValueError("fit state requires initial_root_translations")
    if result.initial_poses_6d is None:
        raise ValueError("fit state requires initial_poses_6d")
    tensors = {
        "beta": np.asarray(result.beta, dtype=np.float64),
        "target_beta": np.asarray(target_beta, dtype=np.float64),
        "initial_beta": np.asarray(result.initial_beta, dtype=np.float64),
        "root_translations": np.asarray(result.root_translations, dtype=np.float64),
        "initial_root_translations": np.asarray(
            result.initial_root_translations,
            dtype=np.float64,
        ),
        "poses_6d": np.asarray(result.poses_6d, dtype=np.float64),
        "initial_poses_6d": np.asarray(result.initial_poses_6d, dtype=np.float64),
    }
    if set(tensors) != set(_FIT_STATE_TENSOR_KEYS):
        raise ValueError("fit state tensors are incomplete")
    return tensors


def _load_fit_state_safetensors(path: Path, *, metadata: dict[str, str]) -> FitState:
    with safe_open(path, framework="numpy") as handle:
        tensors = {
            key: np.asarray(handle.get_tensor(key), dtype=np.float64).copy()
            for key in _FIT_STATE_TENSOR_KEYS
        }
    return FitState(
        path=path,
        source_format="safetensors",
        metadata=dict(metadata),
        **tensors,
    )


def _load_legacy_fit_state_npz(
    path: Path,
    *,
    mode: str,
    expected_frame_count: int,
    expected_joint_count: int,
) -> FitState:
    with np.load(path, allow_pickle=False) as data:
        missing = set(_FIT_STATE_TENSOR_KEYS).difference(data.files)
        if missing:
            raise ValueError(f"legacy fit_state.npz is missing fields: {sorted(missing)}")
        state_mode = str(np.asarray(data["mode"]).item()) if "mode" in data.files else ""
        tensors = {
            key: np.asarray(data[key], dtype=np.float64).copy()
            for key in _FIT_STATE_TENSOR_KEYS
        }
    if state_mode and state_mode != mode:
        raise ValueError(f"legacy fit_state.npz mode {state_mode!r} does not match {mode!r}")
    _validate_fit_state_tensor_shapes(
        tensors,
        expected_frame_count=expected_frame_count,
        expected_joint_count=expected_joint_count,
    )
    return FitState(
        path=path,
        source_format="npz",
        metadata={
            "schema_version": "legacy_npz",
            "mode": state_mode or mode,
            "frame_count": str(expected_frame_count),
            "joint_count": str(expected_joint_count),
        },
        **tensors,
    )


def _validate_fit_state_tensor_shapes(
    tensors: Mapping[str, NDArray[np.float64]],
    *,
    expected_frame_count: int,
    expected_joint_count: int,
) -> None:
    missing = set(_FIT_STATE_TENSOR_KEYS).difference(tensors)
    if missing:
        raise ValueError(f"fit state is missing tensors: {sorted(missing)}")
    beta_shape = tensors["beta"].shape
    if len(beta_shape) != 1:
        raise ValueError("fit state beta must be a vector")
    if tensors["target_beta"].shape != beta_shape or tensors["initial_beta"].shape != beta_shape:
        raise ValueError("fit state beta vectors must have matching shapes")
    expected_translation_shape = (expected_frame_count, 3)
    if tensors["root_translations"].shape != expected_translation_shape:
        raise ValueError("fit state root_translations shape mismatch")
    if tensors["initial_root_translations"].shape != expected_translation_shape:
        raise ValueError("fit state initial_root_translations shape mismatch")
    expected_pose_shape = (expected_frame_count, expected_joint_count, 6)
    if tensors["poses_6d"].shape != expected_pose_shape:
        raise ValueError("fit state poses_6d shape mismatch")
    if tensors["initial_poses_6d"].shape != expected_pose_shape:
        raise ValueError("fit state initial_poses_6d shape mismatch")
    for key, tensor in tensors.items():
        if tensor.size == 0 or not np.isfinite(tensor).all():
            raise ValueError(f"fit state tensor {key!r} must be finite and non-empty")


def _validate_checkpoint(problem: FitProblem, checkpoint: FitCheckpoint) -> None:
    frame_count = len(problem.observations)
    joint_count = problem.template.joints_rest.shape[0]
    _validate_array("checkpoint.beta", checkpoint.beta, problem.beta.shape)
    _validate_array("checkpoint.root_translations", checkpoint.root_translations, (frame_count, 3))
    _validate_array("checkpoint.poses_6d", checkpoint.poses_6d, (frame_count, joint_count, 6))
    _validate_array("checkpoint.initial_beta", checkpoint.initial_beta, problem.beta.shape)
    _validate_array(
        "checkpoint.initial_root_translations",
        checkpoint.initial_root_translations,
        (frame_count, 3),
    )
    if not np.isfinite(checkpoint.initial_loss):
        raise ValueError("checkpoint.initial_loss must be finite")
    for key, values in _copy_loss_history(checkpoint.loss_history).items():
        if not np.all(np.isfinite(values)):
            raise ValueError(f"checkpoint.loss_history[{key!r}] must be finite")


def _validate_array(name: str, value: NDArray[np.floating], shape: tuple[int, ...]) -> None:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")


def _load_tensor_ref(ref: TensorRef, key: str) -> NDArray[np.float64]:
    with np.load(ref.path) as data:
        if key not in data:
            raise ValueError(f"{ref.path} is missing array {key!r}")
        return np.asarray(data[key], dtype=np.float64)


def _optimize_frame_translation(
    vertices: NDArray[np.floating],
    joints: NDArray[np.floating],
    faces: NDArray[np.integer],
    template: CatTemplate,
    observation: FitObservation,
    initial_translation: NDArray[np.floating],
    config: OptimizerConfig,
    history: list[float],
) -> NDArray[np.float64]:
    translation = np.asarray(initial_translation, dtype=np.float64).copy()
    lr = config.lr

    def objective(value: NDArray[np.float64]) -> float:
        return frame_loss(vertices, joints, faces, observation, value, config, template=template)

    def components(value: NDArray[np.float64]) -> LossComponents:
        return frame_loss_components(
            vertices,
            joints,
            faces,
            observation,
            value,
            config,
            template=template,
        )

    initial_components = components(translation)
    current = initial_components.total
    history.append(current)
    for _ in range(config.camera_root_steps):
        gradient = _finite_difference_gradient(objective, translation, config.finite_difference_eps)
        if not np.all(np.isfinite(gradient)):
            break
        accepted = False
        step_lr = lr
        while step_lr >= config.min_lr:
            candidate = translation - step_lr * gradient
            candidate_components = components(candidate)
            candidate_loss = candidate_components.total
            components_not_worse = (
                candidate_components.keypoint <= initial_components.keypoint + 1e-12
                and candidate_components.silhouette
                <= initial_components.silhouette + 1e-12
                and candidate_components.feature_anchor
                <= initial_components.feature_anchor + 1e-12
            )
            if (
                np.isfinite(candidate_loss)
                and candidate_loss <= current
                and components_not_worse
            ):
                translation = candidate
                current = candidate_loss
                lr = min(step_lr * 1.05, config.lr)
                accepted = True
                break
            step_lr *= 0.5
        history.append(current)
        if not accepted:
            break
    return translation


def _finite_difference_gradient(
    objective: Callable[[NDArray[np.float64]], float],
    vector: NDArray[np.float64],
    eps: float,
) -> NDArray[np.float64]:
    gradient = np.zeros_like(vector, dtype=np.float64)
    for index in range(vector.shape[0]):
        plus = vector.copy()
        minus = vector.copy()
        plus[index] += eps
        minus[index] -= eps
        gradient[index] = (objective(plus) - objective(minus)) / (2.0 * eps)
    return gradient


def _optimize_shared_beta(
    problem: FitProblem,
    root_translations: NDArray[np.float64],
    initial_beta: NDArray[np.float64],
    config: OptimizerConfig,
    history: list[float],
) -> NDArray[np.float64]:
    beta = initial_beta.copy()
    lr = config.shape_lr

    def objective(value: NDArray[np.float64]) -> float:
        return total_loss(problem, root_translations, config, beta=value)

    current = objective(beta)
    history.append(current)
    for _ in range(max(config.shape_steps, 0)):
        gradient = _finite_difference_gradient(objective, beta, config.beta_finite_difference_eps)
        if not np.all(np.isfinite(gradient)):
            break
        accepted = False
        step_lr = lr
        while step_lr >= config.min_lr:
            candidate = beta - step_lr * gradient
            candidate_loss = objective(candidate)
            if np.isfinite(candidate_loss) and candidate_loss <= current:
                beta = candidate
                current = candidate_loss
                lr = min(step_lr * 1.05, config.shape_lr)
                accepted = True
                break
            step_lr *= 0.5
        history.append(current)
        if not accepted:
            break
    return beta


def _initial_root_translations(
    problem: FitProblem,
    initial_root_translations: NDArray[np.floating] | None,
) -> NDArray[np.float64]:
    frame_count = len(problem.observations)
    if initial_root_translations is None:
        return np.zeros((frame_count, 3), dtype=np.float64)
    translations = np.asarray(initial_root_translations, dtype=np.float64)
    if translations.shape != (frame_count, 3):
        raise ValueError(f"initial_root_translations must have shape ({frame_count}, 3)")
    return translations.copy()


def _initial_beta(
    problem: FitProblem,
    initial_beta: NDArray[np.floating] | None,
) -> NDArray[np.float64]:
    if initial_beta is None:
        return np.zeros_like(problem.beta, dtype=np.float64)
    beta = np.asarray(initial_beta, dtype=np.float64)
    if beta.shape != problem.beta.shape:
        raise ValueError(f"initial_beta must have shape {problem.beta.shape}")
    return beta.copy()


def _initial_poses_6d(
    problem: FitProblem,
    initial_poses_6d: NDArray[np.floating] | None,
) -> NDArray[np.float64]:
    frame_count = len(problem.observations)
    joint_count = problem.template.joints_rest.shape[0]
    if initial_poses_6d is None:
        return np.tile(_identity_rotation_6d(), (frame_count, joint_count, 1))
    poses = np.asarray(initial_poses_6d, dtype=np.float64)
    expected_shape = (frame_count, joint_count, 6)
    if poses.shape != expected_shape:
        raise ValueError(f"initial_poses_6d must have shape {expected_shape}")
    if not np.all(np.isfinite(poses)):
        raise ValueError("initial_poses_6d must be finite")
    return poses.copy()


def _identity_rotation_6d() -> NDArray[np.float64]:
    return np.asarray([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float64)
