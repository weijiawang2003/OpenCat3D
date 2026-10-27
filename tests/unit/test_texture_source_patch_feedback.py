import json

import numpy as np

from opencat3d.fitting.optimizer import FitObservation
from opencat3d.schemas import PerspectiveCameraState
from opencat3d.textures.basecolor import (
    _bounded_unique_rgb_count,
    _eye_source_detail_reinforcement_alpha,
    _face_feature_palette_from_observation,
    _FaceFeaturePalette,
    _feature_anchor_targets_from_texture_frame,
    _feature_components,
    _iris_source_luma_core_mask,
    _localized_feature_component_mask,
    _merge_best_source_patches,
    _paint_feature_visibility_reinforcement,
    _paint_patch_into_mask,
    _RawTextureViewQuality,
    _sample_dark_pupil_color,
    _sample_iris_color,
    _source_observed_mask_from_local_mask,
    _source_patch_destination_sample_count,
    _source_patch_direct_uploaded_photo_candidate,
    _source_patch_feedback_from_previous_diagnostics,
    _source_patch_feedback_score,
    _source_patch_footprint,
    _source_patch_frontal_photo_priority,
    _source_patch_frontal_photo_priority_diagnostic,
    _source_patch_quality_score,
    _source_patch_unique_rgb_count,
    _source_shape_mask_from_local_mask,
)


def test_feature_anchor_targets_from_texture_frame_loads_sidecar(tmp_path) -> None:
    target_path = tmp_path / "frame_a" / "face_anchor_targets.json"
    target_path.parent.mkdir()
    target_path.write_text(
        json.dumps(
            {
                "targets": {
                    "left_eye_center": {
                        "x": 24.0,
                        "y": 31.0,
                        "confidence": 0.75,
                    },
                    "right_eye_center": {
                        "x": 62.0,
                        "y": 32.0,
                        "confidence": 1.25,
                    },
                    "nose_tip": {
                        "x": 44.0,
                        "y": 52.0,
                        "confidence": 0.0,
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    targets = _feature_anchor_targets_from_texture_frame(
        {"feature_anchors_xyc_path": "frame_a/face_anchor_targets.json"},
        observations_dir=tmp_path,
    )

    assert targets == {
        "left_eye_center": (24.0, 31.0, 0.75),
        "right_eye_center": (62.0, 32.0, 1.0),
    }


def test_face_feature_palette_uses_observed_anchor_guided_patches() -> None:
    height = 96
    width = 96
    image = np.full((height, width, 3), 0.18, dtype=np.float32)
    mask = np.zeros((height, width), dtype=np.bool_)
    mask[10:86, 10:86] = True
    image[mask] = np.asarray([0.93, 0.92, 0.86], dtype=np.float32)
    image[31:39, 28:36] = np.asarray([0.14, 0.08, 0.03], dtype=np.float32)
    image[31:39, 60:68] = np.asarray([0.14, 0.08, 0.03], dtype=np.float32)
    image[33:38, 31:35] = np.asarray([0.56, 0.39, 0.13], dtype=np.float32)
    image[33:38, 63:67] = np.asarray([0.56, 0.39, 0.13], dtype=np.float32)
    image[52:60, 45:53] = np.asarray([0.98, 0.42, 0.48], dtype=np.float32)
    image[64:75, 34:63] = np.asarray([0.98, 0.97, 0.91], dtype=np.float32)
    observation = FitObservation(
        frame_id="anchored_face",
        camera=PerspectiveCameraState(
            image_width=width,
            image_height=height,
            fx=100.0,
            fy=100.0,
            cx=width * 0.5,
            cy=height * 0.5,
            translation_world_to_camera=[0.0, 0.0, 4.0],
        ),
        keypoints_xyc=np.zeros((11, 3), dtype=np.float64),
        hard_mask=mask.astype(np.float32),
        bbox_xyxy=np.asarray([10.0, 10.0, 86.0, 86.0], dtype=np.float64),
        feature_anchors_xyc={
            "left_eye_center": (32.0, 35.0, 0.82),
            "right_eye_center": (64.0, 35.0, 0.84),
            "nose_tip": (49.0, 56.0, 0.88),
            "muzzle_center": (49.0, 69.0, 0.78),
        },
    )

    palette = _face_feature_palette_from_observation(
        observation,
        image,
        mask,
        metadata={
            "media_type": "photo",
            "quality": {
                "raw_frontal_face_confidence": 0.95,
                "raw_keypoint_view_weight": 0.95,
            },
        },
        fallback_eye=np.asarray([20, 18, 16], dtype=np.uint8),
        fallback_nose=np.asarray([180, 120, 125], dtype=np.uint8),
        fallback_muzzle=np.asarray([220, 220, 210], dtype=np.uint8),
    )

    assert palette.source_patch_observed_anchor_guided["left_eye"] is True
    assert palette.source_patch_observed_anchor_guided["right_eye"] is True
    assert palette.source_patch_observed_anchor_guided["nose"] is True
    assert palette.source_patch_observed_anchor_guided["muzzle"] is True
    assert palette.source_patch_landmark_confidences["nose"] == 0.88
    assert palette.source_patch_landmark_confidences["muzzle"] == 0.78
    assert palette.source_patch_localization_selected_counts["nose"] > 0
    assert palette.source_patch_localization_selected_counts["muzzle"] > 0
    assert palette.pupil_rgb is not None
    assert float(np.mean(palette.pupil_rgb)) + 30.0 < float(np.mean(palette.iris_rgb))


def test_dark_pupil_color_uses_observed_eye_core_pixels() -> None:
    pixels = np.asarray(
        [
            [0.54, 0.42, 0.17],
            [0.50, 0.38, 0.14],
            [0.06, 0.04, 0.03],
            [0.04, 0.03, 0.02],
            [0.62, 0.48, 0.21],
            [0.58, 0.44, 0.19],
        ],
        dtype=np.float32,
    )

    sampled = _sample_dark_pupil_color(
        pixels,
        np.asarray([90, 75, 60], dtype=np.uint8),
    )

    assert int(np.max(sampled)) < 20


def test_iris_color_prefers_observed_chromatic_iris_pixels() -> None:
    pixels = np.asarray(
        [
            [0.08, 0.06, 0.04],
            [0.20, 0.18, 0.16],
            [0.50, 0.46, 0.40],
            [0.62, 0.52, 0.22],
            [0.66, 0.55, 0.24],
            [0.70, 0.58, 0.25],
            [0.82, 0.80, 0.76],
        ],
        dtype=np.float32,
    )

    sampled = _sample_iris_color(
        pixels,
        np.asarray([24, 18, 16], dtype=np.uint8),
    )

    assert sampled[0] > sampled[2]
    assert sampled[1] > sampled[2]
    assert int(np.mean(sampled)) > 100


def _palette_with_left_eye_patch(
    *,
    frame_id: str,
    quality_score: float,
    frontal_face_confidence: float,
    view_weight: float,
    landmark_confidence: float = 0.1,
    frontal_photo_priority: float = 0.0,
    direct_photo_candidate: bool = False,
    selected_count: int = 64,
    candidate_count: int = 64,
    allowed_source_patch_sample_keys: tuple[str, ...] = (),
) -> _FaceFeaturePalette:
    patch = np.full((8, 8, 3), 120, dtype=np.uint8)
    patch_mask = np.full((8, 8), 255, dtype=np.uint8)
    sample_count = max(int(selected_count), 1)
    return _FaceFeaturePalette(
        source_frame_id=frame_id,
        source_score=0.0,
        eye_rgb=np.asarray([20, 18, 16], dtype=np.uint8),
        iris_rgb=np.asarray([42, 38, 30], dtype=np.uint8),
        nose_rgb=np.asarray([180, 120, 125], dtype=np.uint8),
        muzzle_rgb=np.asarray([220, 220, 210], dtype=np.uint8),
        ear_inner_rgb=np.asarray([180, 120, 125], dtype=np.uint8),
        mouth_rgb=np.asarray([20, 18, 16], dtype=np.uint8),
        sample_counts={"left_eye": sample_count},
        source_patches={"left_eye": patch},
        source_patch_masks={"left_eye": patch_mask},
        source_patch_mask_observed_counts={"left_eye": sample_count},
        source_patch_mask_densified_counts={"left_eye": sample_count},
        source_patch_mask_densification_gains={"left_eye": 1.0},
        source_patch_localization_candidate_counts={
            "left_eye": max(int(candidate_count), 1),
        },
        source_patch_localization_selected_counts={"left_eye": sample_count},
        source_patch_localization_fractions={
            "left_eye": sample_count / max(int(candidate_count), 1),
        },
        source_patch_landmark_confidences={"left_eye": landmark_confidence},
        source_patch_scores={"left_eye": quality_score},
        source_patch_frame_ids={"left_eye": frame_id},
        source_patch_frontal_face_confidences={
            "left_eye": frontal_face_confidence,
        },
        source_patch_view_weights={"left_eye": view_weight},
        source_patch_view_labels={"left_eye": f"{frame_id}_view"},
        source_patch_view_weighted_ranks={"left_eye": 0.0},
        source_patch_feedback_components={},
        source_patch_feedback_scores={},
        source_patch_frontal_photo_priorities={
            "left_eye": frontal_photo_priority,
        },
        source_patch_direct_photo_candidates={
            "left_eye": direct_photo_candidate,
        },
        allowed_source_patch_sample_keys=allowed_source_patch_sample_keys,
    )


def _palette_with_iris_patch(
    *,
    frame_id: str,
    sample_key: str,
    quality_score: float,
    frontal_face_confidence: float,
    view_weight: float,
    landmark_confidence: float,
    frontal_photo_priority: float = 0.0,
    direct_photo_candidate: bool = False,
    sample_count: int = 1024,
    detailed: bool = True,
) -> _FaceFeaturePalette:
    patch = np.full((24, 24, 3), [132, 108, 58], dtype=np.uint8)
    if detailed:
        patch[::2, :, :] = [74, 58, 28]
        patch[1::2, :, :] = [196, 160, 74]
    patch_mask = np.full((24, 24), 255, dtype=np.uint8)
    return _FaceFeaturePalette(
        source_frame_id=frame_id,
        source_score=0.0,
        eye_rgb=np.asarray([35, 28, 24], dtype=np.uint8),
        iris_rgb=np.asarray([132, 108, 58], dtype=np.uint8),
        nose_rgb=np.asarray([180, 120, 125], dtype=np.uint8),
        muzzle_rgb=np.asarray([220, 220, 210], dtype=np.uint8),
        ear_inner_rgb=np.asarray([180, 120, 125], dtype=np.uint8),
        mouth_rgb=np.asarray([20, 18, 16], dtype=np.uint8),
        sample_counts={sample_key: sample_count},
        source_patches={sample_key: patch},
        source_patch_masks={sample_key: patch_mask},
        source_patch_mask_observed_counts={sample_key: sample_count},
        source_patch_mask_densified_counts={sample_key: sample_count},
        source_patch_mask_densification_gains={sample_key: 1.0},
        source_patch_localization_candidate_counts={sample_key: sample_count},
        source_patch_localization_selected_counts={sample_key: sample_count},
        source_patch_localization_fractions={sample_key: 1.0},
        source_patch_landmark_confidences={sample_key: landmark_confidence},
        source_patch_scores={sample_key: quality_score},
        source_patch_frame_ids={sample_key: frame_id},
        source_patch_frontal_face_confidences={sample_key: frontal_face_confidence},
        source_patch_view_weights={sample_key: view_weight},
        source_patch_view_labels={sample_key: f"{frame_id}_view"},
        source_patch_view_weighted_ranks={sample_key: 0.0},
        source_patch_feedback_components={},
        source_patch_feedback_scores={},
        source_patch_frontal_photo_priorities={
            sample_key: frontal_photo_priority,
        },
        source_patch_direct_photo_candidates={
            sample_key: direct_photo_candidate,
        },
    )


def _palette_with_pupil_patch(
    *,
    frame_id: str,
    sample_key: str,
    quality_score: float,
    frontal_face_confidence: float,
    view_weight: float,
    landmark_confidence: float,
    frontal_photo_priority: float = 0.0,
    direct_photo_candidate: bool = False,
    sample_count: int = 220,
    selected_count: int = 220,
    candidate_count: int = 240,
    dark: bool = True,
) -> _FaceFeaturePalette:
    patch = np.full((32, 32, 3), [26, 20, 18] if dark else [84, 78, 68], dtype=np.uint8)
    if dark:
        patch[12:20, 12:20] = [10, 8, 7]
    else:
        patch[::4, :, :] = [42, 38, 32]
    patch_mask = np.full((32, 32), 255, dtype=np.uint8)
    return _FaceFeaturePalette(
        source_frame_id=frame_id,
        source_score=0.0,
        eye_rgb=np.asarray([24, 18, 16], dtype=np.uint8),
        iris_rgb=np.asarray([132, 108, 58], dtype=np.uint8),
        nose_rgb=np.asarray([180, 120, 125], dtype=np.uint8),
        muzzle_rgb=np.asarray([220, 220, 210], dtype=np.uint8),
        ear_inner_rgb=np.asarray([180, 120, 125], dtype=np.uint8),
        mouth_rgb=np.asarray([20, 18, 16], dtype=np.uint8),
        sample_counts={sample_key: sample_count},
        source_patches={sample_key: patch},
        source_patch_masks={sample_key: patch_mask},
        source_patch_mask_observed_counts={sample_key: sample_count},
        source_patch_mask_densified_counts={sample_key: sample_count},
        source_patch_mask_densification_gains={sample_key: 1.0},
        source_patch_localization_candidate_counts={sample_key: candidate_count},
        source_patch_localization_selected_counts={sample_key: selected_count},
        source_patch_localization_fractions={
            sample_key: selected_count / max(candidate_count, 1),
        },
        source_patch_landmark_confidences={sample_key: landmark_confidence},
        source_patch_scores={sample_key: quality_score},
        source_patch_frame_ids={sample_key: frame_id},
        source_patch_frontal_face_confidences={sample_key: frontal_face_confidence},
        source_patch_view_weights={sample_key: view_weight},
        source_patch_view_labels={sample_key: f"{frame_id}_view"},
        source_patch_view_weighted_ranks={sample_key: 0.0},
        source_patch_feedback_components={},
        source_patch_feedback_scores={},
        source_patch_frontal_photo_priorities={
            sample_key: frontal_photo_priority,
        },
        source_patch_direct_photo_candidates={
            sample_key: direct_photo_candidate,
        },
        pupil_rgb=np.asarray([10, 8, 7] if dark else [42, 38, 32], dtype=np.uint8),
    )


def _palette_with_nose_patch(
    *,
    frame_id: str,
    quality_score: float,
    frontal_face_confidence: float,
    view_weight: float,
    nose_rgb: tuple[int, int, int],
    muzzle_rgb: tuple[int, int, int],
    landmark_confidence: float = 0.5,
    frontal_photo_priority: float = 0.0,
    direct_photo_candidate: bool = False,
) -> _FaceFeaturePalette:
    patch = np.full((8, 8, 3), nose_rgb, dtype=np.uint8)
    patch_mask = np.full((8, 8), 255, dtype=np.uint8)
    return _FaceFeaturePalette(
        source_frame_id=frame_id,
        source_score=0.0,
        eye_rgb=np.asarray([35, 28, 24], dtype=np.uint8),
        iris_rgb=np.asarray([72, 64, 48], dtype=np.uint8),
        nose_rgb=np.asarray(nose_rgb, dtype=np.uint8),
        muzzle_rgb=np.asarray(muzzle_rgb, dtype=np.uint8),
        ear_inner_rgb=np.asarray([190, 120, 125], dtype=np.uint8),
        mouth_rgb=np.asarray([45, 35, 30], dtype=np.uint8),
        sample_counts={"nose": 64},
        source_patches={"nose": patch},
        source_patch_masks={"nose": patch_mask},
        source_patch_mask_observed_counts={"nose": 64},
        source_patch_mask_densified_counts={"nose": 64},
        source_patch_mask_densification_gains={"nose": 1.0},
        source_patch_localization_candidate_counts={"nose": 64},
        source_patch_localization_selected_counts={"nose": 64},
        source_patch_localization_fractions={"nose": 1.0},
        source_patch_landmark_confidences={"nose": landmark_confidence},
        source_patch_scores={"nose": quality_score},
        source_patch_frame_ids={"nose": frame_id},
        source_patch_frontal_face_confidences={
            "nose": frontal_face_confidence,
        },
        source_patch_view_weights={"nose": view_weight},
        source_patch_view_labels={"nose": f"{frame_id}_view"},
        source_patch_view_weighted_ranks={"nose": 0.0},
        source_patch_feedback_components={},
        source_patch_feedback_scores={},
        source_patch_frontal_photo_priorities={"nose": frontal_photo_priority},
        source_patch_direct_photo_candidates={"nose": direct_photo_candidate},
    )


def _palette_with_muzzle_patch(
    *,
    frame_id: str,
    quality_score: float,
    frontal_face_confidence: float,
    view_weight: float,
    sample_count: int,
    patch_shape: tuple[int, int],
    frontal_photo_priority: float = 0.0,
    direct_photo_candidate: bool = False,
) -> _FaceFeaturePalette:
    height, width = patch_shape
    patch = np.full((height, width, 3), 220, dtype=np.uint8)
    patch[::2, :, :] = 190
    patch_mask = np.full((height, width), 255, dtype=np.uint8)
    return _FaceFeaturePalette(
        source_frame_id=frame_id,
        source_score=0.0,
        eye_rgb=np.asarray([35, 28, 24], dtype=np.uint8),
        iris_rgb=np.asarray([72, 64, 48], dtype=np.uint8),
        nose_rgb=np.asarray([185, 118, 124], dtype=np.uint8),
        muzzle_rgb=np.asarray([220, 220, 210], dtype=np.uint8),
        ear_inner_rgb=np.asarray([190, 120, 125], dtype=np.uint8),
        mouth_rgb=np.asarray([45, 35, 30], dtype=np.uint8),
        sample_counts={"muzzle": sample_count},
        source_patches={"muzzle": patch},
        source_patch_masks={"muzzle": patch_mask},
        source_patch_mask_observed_counts={"muzzle": sample_count},
        source_patch_mask_densified_counts={"muzzle": sample_count},
        source_patch_mask_densification_gains={"muzzle": 1.0},
        source_patch_localization_candidate_counts={"muzzle": sample_count},
        source_patch_localization_selected_counts={"muzzle": sample_count},
        source_patch_localization_fractions={"muzzle": 1.0},
        source_patch_landmark_confidences={"muzzle": 0.7},
        source_patch_scores={"muzzle": quality_score},
        source_patch_frame_ids={"muzzle": frame_id},
        source_patch_frontal_face_confidences={"muzzle": frontal_face_confidence},
        source_patch_view_weights={"muzzle": view_weight},
        source_patch_view_labels={"muzzle": f"{frame_id}_view"},
        source_patch_view_weighted_ranks={"muzzle": 0.0},
        source_patch_feedback_components={},
        source_patch_feedback_scores={},
        source_patch_frontal_photo_priorities={
            "muzzle": frontal_photo_priority,
        },
        source_patch_direct_photo_candidates={
            "muzzle": direct_photo_candidate,
        },
    )


def _palette_with_mouth_patch(
    *,
    frame_id: str,
    quality_score: float,
    frontal_face_confidence: float,
    view_weight: float,
    mouth_rgb: tuple[int, int, int],
    patch_rgb: tuple[int, int, int],
    sample_count: int,
) -> _FaceFeaturePalette:
    patch = np.full((24, 24, 3), patch_rgb, dtype=np.uint8)
    patch_mask = np.full((24, 24), 255, dtype=np.uint8)
    return _FaceFeaturePalette(
        source_frame_id=frame_id,
        source_score=0.0,
        eye_rgb=np.asarray([35, 28, 24], dtype=np.uint8),
        iris_rgb=np.asarray([72, 64, 48], dtype=np.uint8),
        nose_rgb=np.asarray([185, 118, 124], dtype=np.uint8),
        muzzle_rgb=np.asarray([220, 220, 210], dtype=np.uint8),
        ear_inner_rgb=np.asarray([190, 120, 125], dtype=np.uint8),
        mouth_rgb=np.asarray(mouth_rgb, dtype=np.uint8),
        sample_counts={"mouth": sample_count},
        source_patches={"mouth": patch},
        source_patch_masks={"mouth": patch_mask},
        source_patch_mask_observed_counts={"mouth": sample_count},
        source_patch_mask_densified_counts={"mouth": sample_count},
        source_patch_mask_densification_gains={"mouth": 1.0},
        source_patch_localization_candidate_counts={"mouth": sample_count},
        source_patch_localization_selected_counts={"mouth": sample_count},
        source_patch_localization_fractions={"mouth": 1.0},
        source_patch_landmark_confidences={"mouth": 0.7},
        source_patch_scores={"mouth": quality_score},
        source_patch_frame_ids={"mouth": frame_id},
        source_patch_frontal_face_confidences={"mouth": frontal_face_confidence},
        source_patch_view_weights={"mouth": view_weight},
        source_patch_view_labels={"mouth": f"{frame_id}_view"},
        source_patch_view_weighted_ranks={"mouth": 0.0},
        source_patch_feedback_components={},
        source_patch_feedback_scores={},
    )


def _palette_with_ear_inner_patch(
    *,
    frame_id: str,
    quality_score: float,
    frontal_face_confidence: float,
    view_weight: float,
    sample_count: int,
    detailed: bool,
    frontal_photo_priority: float = 0.0,
    direct_photo_candidate: bool = False,
) -> _FaceFeaturePalette:
    patch = np.full((32, 32, 3), [190, 136, 136], dtype=np.uint8)
    if detailed:
        patch[::2, :, :] = [118, 70, 75]
        patch[1::2, :, :] = [226, 180, 176]
    patch_mask = np.full((32, 32), 255, dtype=np.uint8)
    return _FaceFeaturePalette(
        source_frame_id=frame_id,
        source_score=0.0,
        eye_rgb=np.asarray([35, 28, 24], dtype=np.uint8),
        iris_rgb=np.asarray([72, 64, 48], dtype=np.uint8),
        nose_rgb=np.asarray([185, 118, 124], dtype=np.uint8),
        muzzle_rgb=np.asarray([220, 220, 210], dtype=np.uint8),
        ear_inner_rgb=np.asarray([190, 136, 136], dtype=np.uint8),
        mouth_rgb=np.asarray([45, 35, 30], dtype=np.uint8),
        sample_counts={"ear_inner": sample_count},
        source_patches={"ear_inner": patch},
        source_patch_masks={"ear_inner": patch_mask},
        source_patch_mask_observed_counts={"ear_inner": sample_count},
        source_patch_mask_densified_counts={"ear_inner": sample_count},
        source_patch_mask_densification_gains={"ear_inner": 1.0},
        source_patch_localization_candidate_counts={"ear_inner": sample_count},
        source_patch_localization_selected_counts={"ear_inner": sample_count},
        source_patch_localization_fractions={"ear_inner": 1.0},
        source_patch_landmark_confidences={"ear_inner": 0.0},
        source_patch_scores={"ear_inner": quality_score},
        source_patch_frame_ids={"ear_inner": frame_id},
        source_patch_frontal_face_confidences={
            "ear_inner": frontal_face_confidence,
        },
        source_patch_view_weights={"ear_inner": view_weight},
        source_patch_view_labels={"ear_inner": f"{frame_id}_view"},
        source_patch_view_weighted_ranks={"ear_inner": 0.0},
        source_patch_feedback_components={},
        source_patch_feedback_scores={},
        source_patch_frontal_photo_priorities={"ear_inner": frontal_photo_priority},
        source_patch_direct_photo_candidates={
            "ear_inner": direct_photo_candidate,
        },
    )


def test_source_patch_merge_prefers_frontal_face_view_for_eye_patch() -> None:
    side_quality = _palette_with_left_eye_patch(
        frame_id="side",
        quality_score=0.95,
        frontal_face_confidence=0.10,
        view_weight=0.40,
    )
    frontal = _palette_with_left_eye_patch(
        frame_id="frontal",
        quality_score=0.45,
        frontal_face_confidence=0.90,
        view_weight=0.95,
    )

    merged = _merge_best_source_patches(
        side_quality,
        [side_quality, frontal],
        source_patch_feedback={},
    )

    assert merged.source_patch_frame_ids["left_eye"] == "frontal"
    assert merged.source_patch_view_labels["left_eye"] == "frontal_view"
    assert merged.source_patch_view_weights["left_eye"] == 0.95
    assert merged.source_patch_view_weighted_ranks["left_eye"] > 0.0


def test_source_patch_merge_respects_allowed_sample_key_policy() -> None:
    baseline = _palette_with_left_eye_patch(
        frame_id="baseline",
        quality_score=0.45,
        frontal_face_confidence=0.55,
        view_weight=0.65,
    )
    restricted_probe = _palette_with_left_eye_patch(
        frame_id="restricted_probe",
        quality_score=0.98,
        frontal_face_confidence=0.99,
        view_weight=0.99,
        allowed_source_patch_sample_keys=("muzzle",),
    )

    merged = _merge_best_source_patches(
        baseline,
        [baseline, restricted_probe],
        source_patch_feedback={},
    )

    assert merged.source_patch_frame_ids["left_eye"] == "baseline"
    assert all(
        candidate["source_frame_id"] != "restricted_probe"
        for candidate in merged.source_patch_candidate_diagnostics["left_eye"]
    )


def test_source_patch_merge_prefers_richer_localized_eye_support() -> None:
    tiny_frontal = _palette_with_left_eye_patch(
        frame_id="tiny_frontal",
        quality_score=0.75,
        frontal_face_confidence=0.95,
        view_weight=0.96,
        selected_count=12,
        candidate_count=240,
    )
    richer_observed = _palette_with_left_eye_patch(
        frame_id="richer_observed",
        quality_score=0.72,
        frontal_face_confidence=0.62,
        view_weight=0.75,
        selected_count=256,
        candidate_count=256,
    )

    merged = _merge_best_source_patches(
        tiny_frontal,
        [tiny_frontal, richer_observed],
        source_patch_feedback={},
    )

    assert merged.source_patch_frame_ids["left_eye"] == "richer_observed"
    assert merged.source_patch_localization_selected_counts["left_eye"] == 256
    assert merged.source_patch_view_weighted_ranks["left_eye"] > 0.0


def test_source_patch_merge_prefers_direct_frontal_photo_eye_source() -> None:
    old_landmark_source = _palette_with_left_eye_patch(
        frame_id="old_landmark_source",
        quality_score=0.62,
        frontal_face_confidence=0.65,
        view_weight=0.76,
        landmark_confidence=0.55,
        frontal_photo_priority=0.0,
        selected_count=160,
        candidate_count=180,
    )
    direct_frontal_photo = _palette_with_left_eye_patch(
        frame_id="direct_frontal_photo",
        quality_score=0.78,
        frontal_face_confidence=0.82,
        view_weight=0.88,
        landmark_confidence=0.0,
        frontal_photo_priority=0.92,
        selected_count=220,
        candidate_count=240,
    )

    merged = _merge_best_source_patches(
        old_landmark_source,
        [old_landmark_source, direct_frontal_photo],
        source_patch_feedback={},
    )

    assert merged.source_patch_frame_ids["left_eye"] == "direct_frontal_photo"
    assert merged.source_patch_frontal_photo_priorities["left_eye"] == 0.92
    assert merged.source_patch_view_weighted_ranks["left_eye"] > 0.0


def test_source_patch_merge_uses_admitted_texture_only_eye_rank_floor() -> None:
    old_landmark_source = _palette_with_left_eye_patch(
        frame_id="old_landmark_source",
        quality_score=1.0,
        frontal_face_confidence=0.72,
        view_weight=0.82,
        landmark_confidence=0.55,
        selected_count=1024,
        candidate_count=1024,
    )
    admitted_closeup = _palette_with_left_eye_patch(
        frame_id="admitted_closeup",
        quality_score=0.52,
        frontal_face_confidence=0.20,
        view_weight=0.24,
        landmark_confidence=0.20,
        frontal_photo_priority=0.52,
        direct_photo_candidate=True,
        selected_count=1024,
        candidate_count=1024,
    )
    admitted_closeup.source_patch_frontal_photo_priority_diagnostics["left_eye"] = {
        "status": "accepted",
        "rejection_reason": "",
        "priority": 0.52,
        "texture_only_source_patch_candidate": True,
        "texture_only_source_patch_effective_view_applied": True,
        "support_score": 1.0,
        "density_score": 1.0,
        "detail_score": 1.0,
        "quality_score": 0.52,
        "view_score": 0.58,
        "raw_view_score": 0.22,
    }

    merged = _merge_best_source_patches(
        old_landmark_source,
        [old_landmark_source, admitted_closeup],
        source_patch_feedback={},
    )

    assert merged.source_patch_frame_ids["left_eye"] == "admitted_closeup"
    assert merged.source_patch_direct_photo_candidates["left_eye"] is True
    closeup_diagnostics = [
        candidate
        for candidate in merged.source_patch_candidate_diagnostics["left_eye"]
        if candidate["source_frame_id"] == "admitted_closeup"
    ]
    assert closeup_diagnostics[0]["source_patch_texture_only_rank_floor"] > 0.0
    assert closeup_diagnostics[0]["source_patch_texture_only_source_patch_candidate"] is True


def test_frontal_photo_priority_requires_direct_upload_source_name() -> None:
    patch = np.full((24, 24, 3), 128, dtype=np.uint8)
    patch[::2, :, :] = 32
    patch[1::2, :, :] = 224
    view_quality = _RawTextureViewQuality(
        label="frontal_face_candidate",
        frontal_face_confidence=0.88,
        view_weight=0.92,
    )
    common = {
        "sample_key": "left_eye",
        "view_quality": view_quality,
        "patch": patch,
        "selected_count": 512,
        "candidate_count": 512,
        "sample_count": 512,
        "source_quality_score": 0.85,
    }

    direct = _source_patch_frontal_photo_priority(
        metadata={
            "media_type": "photo",
            "source_type": "image",
            "source_media_id": "14.jpg",
        },
        **common,
    )
    auto_staged = _source_patch_frontal_photo_priority(
        metadata={
            "media_type": "photo",
            "source_type": "image",
            "source_media_id": "frame_0004.jpg",
        },
        **common,
    )

    assert direct > 0.0
    assert auto_staged == 0.0


def test_frontal_photo_priority_accepts_admitted_texture_only_source_patch() -> None:
    patch = np.full((24, 24, 3), 128, dtype=np.uint8)
    patch[::2, :, :] = 32
    patch[1::2, :, :] = 224
    view_quality = _RawTextureViewQuality(
        label="vertical_body_candidate",
        frontal_face_confidence=0.20,
        view_weight=0.24,
    )
    admitted_metadata = {
        "media_type": "photo",
        "source_type": "image",
        "source_media_id": "frame_0005.jpg",
        "texture_only_supplemental": True,
        "texture_only_source_patch_allowed": True,
        "texture_only_source_preflight_score": 1.12,
        "texture_only_face_region_projection_protection": {
            "applied": True,
            "broad_projection_excluded": True,
            "source_patch_allowed": True,
        },
    }
    unadmitted_metadata = {
        "media_type": "photo",
        "source_type": "image",
        "source_media_id": "frame_0005.jpg",
    }

    accepted = _source_patch_frontal_photo_priority_diagnostic(
        "left_eye",
        metadata=admitted_metadata,
        view_quality=view_quality,
        patch=patch,
        selected_count=512,
        candidate_count=512,
        sample_count=512,
        source_quality_score=0.82,
    )
    rejected = _source_patch_frontal_photo_priority_diagnostic(
        "left_eye",
        metadata=unadmitted_metadata,
        view_quality=view_quality,
        patch=patch,
        selected_count=512,
        candidate_count=512,
        sample_count=512,
        source_quality_score=0.82,
    )

    assert _source_patch_direct_uploaded_photo_candidate(admitted_metadata) is True
    assert accepted["status"] == "accepted"
    assert accepted["priority"] > 0.0
    assert accepted["direct_photo_candidate"] is True
    assert accepted["texture_only_source_patch_candidate"] is True
    assert accepted["texture_only_source_patch_effective_view_applied"] is True
    assert accepted["raw_view_score"] < 0.50
    assert accepted["view_score"] >= 0.50
    assert rejected["priority"] == 0.0
    assert rejected["rejection_reason"] == "not_direct_uploaded_photo"


def test_frontal_photo_priority_accepts_dense_detailed_iris_closeup() -> None:
    patch = np.full((24, 24, 3), [132, 108, 58], dtype=np.uint8)
    patch[::2, :, :] = [74, 58, 28]
    patch[1::2, :, :] = [196, 160, 74]
    view_quality = _RawTextureViewQuality(
        label="oblique_body_candidate",
        frontal_face_confidence=0.62,
        view_weight=0.66,
    )
    metadata = {
        "media_type": "photo",
        "source_type": "image",
        "source_media_id": "17.jpg",
    }

    accepted = _source_patch_frontal_photo_priority_diagnostic(
        "left_iris",
        metadata=metadata,
        view_quality=view_quality,
        patch=patch,
        selected_count=1024,
        candidate_count=1024,
        sample_count=1024,
        source_quality_score=0.20,
    )
    flat = _source_patch_frontal_photo_priority_diagnostic(
        "left_iris",
        metadata=metadata,
        view_quality=view_quality,
        patch=np.full_like(patch, [132, 108, 58]),
        selected_count=1024,
        candidate_count=1024,
        sample_count=1024,
        source_quality_score=0.20,
    )

    assert accepted["priority"] > 0.0
    assert accepted["status"] == "accepted"
    assert accepted["quality_min"] == 0.18
    assert accepted["detail_score"] >= 0.75
    assert flat["priority"] == 0.0
    assert flat["rejection_reason"] == "detail_below_min"


def test_frontal_photo_priority_accepts_dark_direct_photo_pupil_closeup() -> None:
    patch = np.full((32, 32, 3), [26, 20, 18], dtype=np.uint8)
    patch[12:20, 12:20] = [10, 8, 7]
    view_quality = _RawTextureViewQuality(
        label="frontal_face_candidate",
        frontal_face_confidence=0.82,
        view_weight=0.86,
    )
    metadata = {
        "media_type": "photo",
        "source_type": "image",
        "source_media_id": "14.jpg",
    }

    accepted = _source_patch_frontal_photo_priority_diagnostic(
        "left_pupil",
        metadata=metadata,
        view_quality=view_quality,
        patch=patch,
        selected_count=220,
        candidate_count=240,
        sample_count=220,
        source_quality_score=0.12,
    )
    bright = _source_patch_frontal_photo_priority_diagnostic(
        "left_pupil",
        metadata=metadata,
        view_quality=view_quality,
        patch=np.full_like(patch, [220, 220, 220]),
        selected_count=220,
        candidate_count=240,
        sample_count=220,
        source_quality_score=0.12,
    )

    assert accepted["status"] == "accepted"
    assert accepted["priority"] > 0.0
    assert accepted["quality_min"] == 0.10
    assert accepted["detail_min"] == 0.0
    assert accepted["pupil_darkness_score"] >= 0.55
    assert bright["priority"] == 0.0
    assert bright["rejection_reason"] == "darkness_below_min"


def test_source_patch_merge_prefers_accepted_direct_photo_iris_detail() -> None:
    older_staged_photo = _palette_with_iris_patch(
        frame_id="older_staged_photo",
        sample_key="left_iris",
        quality_score=0.71,
        frontal_face_confidence=0.79,
        view_weight=0.80,
        landmark_confidence=0.87,
        frontal_photo_priority=0.0,
        direct_photo_candidate=False,
        sample_count=512,
        detailed=True,
    )
    accepted_direct_photo = _palette_with_iris_patch(
        frame_id="accepted_direct_photo",
        sample_key="left_iris",
        quality_score=0.20,
        frontal_face_confidence=0.62,
        view_weight=0.66,
        landmark_confidence=0.93,
        frontal_photo_priority=0.52,
        direct_photo_candidate=True,
        sample_count=2048,
        detailed=True,
    )

    merged = _merge_best_source_patches(
        older_staged_photo,
        [older_staged_photo, accepted_direct_photo],
        source_patch_feedback={},
    )

    assert merged.source_patch_frame_ids["left_iris"] == "accepted_direct_photo"
    assert merged.source_patch_direct_photo_candidates["left_iris"] is True
    assert merged.source_patch_frontal_photo_priorities["left_iris"] == 0.52


def test_source_patch_merge_prefers_accepted_direct_photo_pupil_core() -> None:
    staged_pupil = _palette_with_pupil_patch(
        frame_id="staged_pupil",
        sample_key="left_pupil",
        quality_score=0.53,
        frontal_face_confidence=0.75,
        view_weight=0.84,
        landmark_confidence=0.86,
        frontal_photo_priority=0.0,
        direct_photo_candidate=False,
        sample_count=220,
        selected_count=220,
        candidate_count=236,
        dark=False,
    )
    direct_pupil = _palette_with_pupil_patch(
        frame_id="direct_uploaded_pupil",
        sample_key="left_pupil",
        quality_score=0.12,
        frontal_face_confidence=0.82,
        view_weight=0.86,
        landmark_confidence=0.86,
        frontal_photo_priority=0.61,
        direct_photo_candidate=True,
        sample_count=220,
        selected_count=220,
        candidate_count=220,
        dark=True,
    )

    merged = _merge_best_source_patches(
        staged_pupil,
        [staged_pupil, direct_pupil],
        source_patch_feedback={},
    )

    assert merged.source_patch_frame_ids["left_pupil"] == "direct_uploaded_pupil"
    assert merged.source_patch_direct_photo_candidates["left_pupil"] is True
    assert merged.source_patch_frontal_photo_priorities["left_pupil"] == 0.61
    assert merged.pupil_rgb is not None
    assert int(np.max(merged.pupil_rgb)) <= 10
    assert merged.source_patch_view_weighted_ranks["left_pupil"] > (
        staged_pupil.source_patch_view_weighted_ranks["left_pupil"]
    )


def test_frontal_photo_priority_accepts_dense_white_muzzle_closeup() -> None:
    patch = np.full((72, 72, 3), 235, dtype=np.uint8)
    patch[::4, :, :] = 231
    view_quality = _RawTextureViewQuality(
        label="frontal_face_candidate",
        frontal_face_confidence=0.74,
        view_weight=0.80,
    )
    common = {
        "sample_key": "muzzle",
        "metadata": {
            "media_type": "photo",
            "source_type": "image",
            "source_media_id": "14.jpg",
        },
        "view_quality": view_quality,
        "patch": patch,
        "sample_count": 2048,
        "source_quality_score": 0.48,
    }

    dense_closeup = _source_patch_frontal_photo_priority(
        selected_count=2048,
        candidate_count=2048,
        **common,
    )
    sparse_closeup = _source_patch_frontal_photo_priority(
        selected_count=64,
        candidate_count=2048,
        **common,
    )

    assert dense_closeup > 0.0
    assert sparse_closeup == 0.0


def test_frontal_photo_priority_diagnostic_reports_ear_inner_quality_gate() -> None:
    patch = np.full((32, 32, 3), [190, 136, 136], dtype=np.uint8)
    patch[::2, :, :] = [118, 70, 75]
    patch[1::2, :, :] = [226, 180, 176]
    view_quality = _RawTextureViewQuality(
        label="frontal_face_candidate",
        frontal_face_confidence=0.62,
        view_weight=0.62,
    )

    diagnostic = _source_patch_frontal_photo_priority_diagnostic(
        "ear_inner",
        metadata={
            "media_type": "photo",
            "source_type": "image",
            "source_media_id": "14.jpg",
        },
        view_quality=view_quality,
        patch=patch,
        selected_count=768,
        candidate_count=768,
        sample_count=768,
        source_quality_score=0.36,
    )

    assert diagnostic["priority"] == 0.0
    assert diagnostic["status"] == "rejected"
    assert diagnostic["rejection_reason"] == "quality_below_min"
    assert diagnostic["direct_photo_candidate"] is True
    assert diagnostic["quality_min"] == 0.45
    assert diagnostic["detail_score"] >= diagnostic["detail_min"]


def test_frontal_photo_priority_accepts_strong_ear_inner_closeup() -> None:
    patch = np.full((32, 32, 3), [190, 136, 136], dtype=np.uint8)
    patch[::2, :, :] = [118, 70, 75]
    patch[1::2, :, :] = [226, 180, 176]
    view_quality = _RawTextureViewQuality(
        label="frontal_face_candidate",
        frontal_face_confidence=0.82,
        view_weight=0.86,
    )

    diagnostic = _source_patch_frontal_photo_priority_diagnostic(
        "ear_inner",
        metadata={
            "media_type": "photo",
            "source_type": "image",
            "source_media_id": "14.jpg",
        },
        view_quality=view_quality,
        patch=patch,
        selected_count=768,
        candidate_count=768,
        sample_count=768,
        source_quality_score=0.36,
    )

    assert diagnostic["priority"] > 0.0
    assert diagnostic["status"] == "accepted"
    assert diagnostic["rejection_reason"] == ""
    assert diagnostic["quality_min"] == 0.35
    assert diagnostic["detail_score"] >= 0.85
    assert diagnostic["support_score"] >= 0.95
    assert diagnostic["density_score"] >= 0.85


def test_source_patch_merge_rejects_low_contrast_frontal_nose_patch() -> None:
    frontal_low_contrast = _palette_with_nose_patch(
        frame_id="frontal_low_contrast",
        quality_score=0.90,
        frontal_face_confidence=0.95,
        view_weight=0.98,
        nose_rgb=(178, 151, 130),
        muzzle_rgb=(180, 156, 136),
    )
    observed_contrast = _palette_with_nose_patch(
        frame_id="observed_contrast",
        quality_score=0.55,
        frontal_face_confidence=0.70,
        view_weight=0.80,
        nose_rgb=(222, 120, 132),
        muzzle_rgb=(235, 230, 214),
    )

    merged = _merge_best_source_patches(
        frontal_low_contrast,
        [frontal_low_contrast, observed_contrast],
        source_patch_feedback={},
    )

    assert merged.source_patch_frame_ids["nose"] == "observed_contrast"
    assert merged.source_patch_view_labels["nose"] == "observed_contrast_view"
    assert merged.source_patch_view_weighted_ranks["nose"] > 0.0


def test_source_patch_merge_prefers_direct_frontal_photo_nose_evidence() -> None:
    oblique_video = _palette_with_nose_patch(
        frame_id="oblique_video",
        quality_score=0.86,
        frontal_face_confidence=0.63,
        view_weight=0.76,
        nose_rgb=(156, 126, 111),
        muzzle_rgb=(188, 183, 184),
        landmark_confidence=0.76,
    )
    direct_frontal_photo = _palette_with_nose_patch(
        frame_id="direct_frontal_photo",
        quality_score=0.53,
        frontal_face_confidence=0.69,
        view_weight=0.80,
        nose_rgb=(218, 142, 154),
        muzzle_rgb=(235, 230, 224),
        landmark_confidence=0.69,
        frontal_photo_priority=0.56,
        direct_photo_candidate=True,
    )

    merged = _merge_best_source_patches(
        oblique_video,
        [oblique_video, direct_frontal_photo],
        source_patch_feedback={},
    )

    assert merged.source_patch_frame_ids["nose"] == "direct_frontal_photo"
    assert merged.source_patch_frontal_photo_priorities["nose"] == 0.56
    assert merged.source_patch_view_weighted_ranks["nose"] > (
        oblique_video.source_patch_view_weighted_ranks["nose"]
    )


def test_source_patch_merge_rejects_low_contrast_direct_photo_nose_evidence() -> None:
    observed_video = _palette_with_nose_patch(
        frame_id="observed_video",
        quality_score=0.86,
        frontal_face_confidence=0.63,
        view_weight=0.76,
        nose_rgb=(156, 126, 111),
        muzzle_rgb=(188, 183, 184),
        landmark_confidence=0.76,
    )
    low_contrast_direct_photo = _palette_with_nose_patch(
        frame_id="low_contrast_direct_photo",
        quality_score=0.53,
        frontal_face_confidence=0.69,
        view_weight=0.80,
        nose_rgb=(164, 143, 137),
        muzzle_rgb=(188, 183, 184),
        landmark_confidence=0.69,
        frontal_photo_priority=0.56,
        direct_photo_candidate=True,
    )

    merged = _merge_best_source_patches(
        observed_video,
        [observed_video, low_contrast_direct_photo],
        source_patch_feedback={},
    )

    assert merged.source_patch_frame_ids["nose"] == "observed_video"
    assert merged.source_patch_frontal_photo_priorities["nose"] == 0.0


def test_source_patch_merge_prefers_dense_broad_face_patch() -> None:
    sparse_closeup = _palette_with_muzzle_patch(
        frame_id="sparse_closeup",
        quality_score=0.62,
        frontal_face_confidence=0.92,
        view_weight=0.95,
        sample_count=256,
        patch_shape=(96, 96),
    )
    dense_observed = _palette_with_muzzle_patch(
        frame_id="dense_observed",
        quality_score=0.70,
        frontal_face_confidence=0.70,
        view_weight=0.80,
        sample_count=1024,
        patch_shape=(72, 72),
    )

    merged = _merge_best_source_patches(
        sparse_closeup,
        [sparse_closeup, dense_observed],
        source_patch_feedback={},
    )

    assert merged.source_patch_frame_ids["muzzle"] == "dense_observed"
    assert merged.source_patch_localization_selected_counts["muzzle"] == 1024
    assert merged.source_patch_view_weighted_ranks["muzzle"] > 0.0


def test_source_patch_merge_rejects_low_view_broad_face_patch() -> None:
    frontal_face = _palette_with_muzzle_patch(
        frame_id="frontal_face",
        quality_score=0.64,
        frontal_face_confidence=0.79,
        view_weight=0.86,
        sample_count=256,
        patch_shape=(72, 72),
    )
    side_body = _palette_with_muzzle_patch(
        frame_id="side_body",
        quality_score=0.70,
        frontal_face_confidence=0.50,
        view_weight=0.68,
        sample_count=1024,
        patch_shape=(72, 72),
    )

    merged = _merge_best_source_patches(
        frontal_face,
        [frontal_face, side_body],
        source_patch_feedback={},
    )

    assert merged.source_patch_frame_ids["muzzle"] == "frontal_face"
    assert merged.source_patch_view_labels["muzzle"] == "frontal_face_view"
    assert merged.source_patch_view_weighted_ranks["muzzle"] > 0.0


def test_source_patch_merge_penalizes_zero_priority_direct_photo_patch() -> None:
    direct_closeup = _palette_with_muzzle_patch(
        frame_id="direct_closeup",
        quality_score=0.635,
        frontal_face_confidence=0.83,
        view_weight=0.83,
        sample_count=256,
        patch_shape=(72, 72),
        direct_photo_candidate=True,
        frontal_photo_priority=0.0,
    )
    video_detail = _palette_with_muzzle_patch(
        frame_id="video_detail",
        quality_score=0.591,
        frontal_face_confidence=0.70,
        view_weight=0.70,
        sample_count=256,
        patch_shape=(72, 72),
    )

    merged = _merge_best_source_patches(
        direct_closeup,
        [direct_closeup, video_detail],
        source_patch_feedback={},
    )

    assert merged.source_patch_frame_ids["muzzle"] == "video_detail"
    assert merged.source_patch_direct_photo_candidates["muzzle"] is False
    direct_diagnostics = [
        candidate
        for candidate in merged.source_patch_candidate_diagnostics["muzzle"]
        if candidate["source_frame_id"] == "direct_closeup"
    ]
    assert direct_diagnostics[0]["source_patch_zero_priority_direct_photo_rank_penalty"] < 1.0


def test_source_patch_merge_prefers_accepted_direct_photo_white_muzzle() -> None:
    older_frontal = _palette_with_muzzle_patch(
        frame_id="older_frontal",
        quality_score=0.51,
        frontal_face_confidence=0.79,
        view_weight=0.80,
        sample_count=1536,
        patch_shape=(72, 72),
    )
    accepted_direct_photo = _palette_with_muzzle_patch(
        frame_id="accepted_direct_photo",
        quality_score=0.48,
        frontal_face_confidence=0.74,
        view_weight=0.78,
        sample_count=1536,
        patch_shape=(72, 72),
        frontal_photo_priority=0.56,
        direct_photo_candidate=True,
    )

    merged = _merge_best_source_patches(
        older_frontal,
        [older_frontal, accepted_direct_photo],
        source_patch_feedback={},
    )

    assert merged.source_patch_frame_ids["muzzle"] == "accepted_direct_photo"
    assert merged.source_patch_direct_photo_candidates["muzzle"] is True
    assert merged.source_patch_frontal_photo_priorities["muzzle"] == 0.56


def test_source_patch_merge_reports_direct_photo_priority_rejection_reason() -> None:
    direct_closeup = _palette_with_muzzle_patch(
        frame_id="direct_closeup",
        quality_score=0.42,
        frontal_face_confidence=0.83,
        view_weight=0.83,
        sample_count=256,
        patch_shape=(72, 72),
        direct_photo_candidate=True,
        frontal_photo_priority=0.0,
    )
    direct_closeup.source_patch_frontal_photo_priority_diagnostics["muzzle"] = {
        "status": "rejected",
        "rejection_reason": "quality_below_min",
        "view_score": 0.83,
        "support_score": 0.72,
        "density_score": 0.91,
        "detail_min": 0.25,
        "quality_min": 0.45,
        "evidence_score": 0.0,
    }
    video_detail = _palette_with_muzzle_patch(
        frame_id="video_detail",
        quality_score=0.591,
        frontal_face_confidence=0.70,
        view_weight=0.70,
        sample_count=256,
        patch_shape=(72, 72),
    )

    merged = _merge_best_source_patches(
        direct_closeup,
        [direct_closeup, video_detail],
        source_patch_feedback={},
    )

    direct_diagnostics = [
        candidate
        for candidate in merged.source_patch_candidate_diagnostics["muzzle"]
        if candidate["source_frame_id"] == "direct_closeup"
    ]
    assert (
        direct_diagnostics[0]["source_patch_frontal_photo_priority_rejection_reason"]
        == "quality_below_min"
    )
    assert direct_diagnostics[0]["source_patch_frontal_photo_priority_quality_min"] == 0.45


def test_source_patch_merge_prefers_detailed_ear_inner_patch() -> None:
    flat_dense = _palette_with_ear_inner_patch(
        frame_id="flat_dense",
        quality_score=0.92,
        frontal_face_confidence=0.82,
        view_weight=0.88,
        sample_count=1024,
        detailed=False,
    )
    detailed_source = _palette_with_ear_inner_patch(
        frame_id="detailed_source",
        quality_score=0.55,
        frontal_face_confidence=0.62,
        view_weight=0.72,
        sample_count=512,
        detailed=True,
    )

    merged = _merge_best_source_patches(
        flat_dense,
        [flat_dense, detailed_source],
        source_patch_feedback={},
    )

    assert merged.source_patch_frame_ids["ear_inner"] == "detailed_source"
    assert merged.source_patch_view_labels["ear_inner"] == "detailed_source_view"
    assert merged.source_patch_view_weighted_ranks["ear_inner"] > 0.0


def test_source_patch_merge_prefers_direct_frontal_photo_ear_inner_evidence() -> None:
    oblique_body = _palette_with_ear_inner_patch(
        frame_id="oblique_body",
        quality_score=0.83,
        frontal_face_confidence=0.44,
        view_weight=0.64,
        sample_count=1024,
        detailed=True,
    )
    direct_frontal_photo = _palette_with_ear_inner_patch(
        frame_id="direct_frontal_photo",
        quality_score=0.55,
        frontal_face_confidence=0.82,
        view_weight=0.86,
        sample_count=768,
        detailed=True,
        frontal_photo_priority=0.62,
        direct_photo_candidate=True,
    )

    merged = _merge_best_source_patches(
        oblique_body,
        [oblique_body, direct_frontal_photo],
        source_patch_feedback={},
    )

    assert merged.source_patch_frame_ids["ear_inner"] == "direct_frontal_photo"
    assert merged.source_patch_direct_photo_candidates["ear_inner"] is True
    assert merged.source_patch_frontal_photo_priorities["ear_inner"] == 0.62


def test_source_patch_merge_prefers_accepted_direct_ear_inner_over_oblique_quality() -> None:
    oblique_body = _palette_with_ear_inner_patch(
        frame_id="oblique_body",
        quality_score=0.83,
        frontal_face_confidence=0.44,
        view_weight=0.64,
        sample_count=1024,
        detailed=True,
    )
    accepted_direct_photo = _palette_with_ear_inner_patch(
        frame_id="accepted_direct_photo",
        quality_score=0.366,
        frontal_face_confidence=0.82,
        view_weight=0.86,
        sample_count=2048,
        detailed=True,
        frontal_photo_priority=0.68,
        direct_photo_candidate=True,
    )

    merged = _merge_best_source_patches(
        oblique_body,
        [oblique_body, accepted_direct_photo],
        source_patch_feedback={},
    )

    assert merged.source_patch_frame_ids["ear_inner"] == "accepted_direct_photo"
    assert merged.source_patch_direct_photo_candidates["ear_inner"] is True
    assert merged.source_patch_frontal_photo_priorities["ear_inner"] == 0.68


def test_source_patch_merge_rejects_green_background_mouth_patch() -> None:
    green_background = _palette_with_mouth_patch(
        frame_id="green_background",
        quality_score=0.95,
        frontal_face_confidence=0.95,
        view_weight=0.98,
        mouth_rgb=(20, 51, 37),
        patch_rgb=(20, 51, 37),
        sample_count=1024,
    )
    observed_mouth = _palette_with_mouth_patch(
        frame_id="observed_mouth",
        quality_score=0.45,
        frontal_face_confidence=0.70,
        view_weight=0.78,
        mouth_rgb=(58, 40, 43),
        patch_rgb=(58, 40, 43),
        sample_count=256,
    )

    merged = _merge_best_source_patches(
        green_background,
        [green_background, observed_mouth],
        source_patch_feedback={},
    )

    assert merged.source_patch_frame_ids["mouth"] == "observed_mouth"
    assert merged.source_patch_view_labels["mouth"] == "observed_mouth_view"
    assert merged.source_patch_view_weighted_ranks["mouth"] > 0.0


def test_source_patch_feedback_uses_previous_diagnostic_components(tmp_path) -> None:
    project_dir = tmp_path / "project"
    quality_dir = project_dir / "outputs" / "quality"
    quality_dir.mkdir(parents=True)
    (quality_dir / "raw_face_texture_diagnostics.json").write_text(
        json.dumps(
            {
                "regions": [
                    {
                        "name": "left_eye",
                        "texture_visual_quality_failure_reasons": ["coverage"],
                        "texture_visual_quality_weakest_component": "source_luma_score",
                    },
                    {
                        "name": "nose",
                        "texture_visual_quality_failure_reasons": [],
                        "texture_visual_quality_weakest_component": "source_luma_score",
                    },
                    {
                        "name": "left_ear_inner",
                        "texture_visual_quality_failure_reasons": [],
                        "texture_visual_quality_weakest_component": "texture_luma_score",
                    },
                    {
                        "name": "right_ear_inner",
                        "texture_visual_quality_failure_reasons": [],
                        "texture_visual_quality_weakest_component": "source_luma_score",
                    },
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    feedback = _source_patch_feedback_from_previous_diagnostics(
        project_dir,
        mode="raw",
    )

    assert feedback == {
        "left_eye": "confident_texel_score",
        "nose": "source_luma_score",
        "ear_inner": "source_luma_score",
    }


def test_source_patch_feedback_score_prefers_detail_and_coverage() -> None:
    flat_patch = np.full((12, 12, 3), 128, dtype=np.uint8)
    detailed_patch = flat_patch.copy()
    detailed_patch[::2, :, :] = 32
    detailed_patch[1::2, :, :] = 224

    assert _source_patch_feedback_score(
        "source_luma_score",
        patch=detailed_patch,
        sample_count=20,
        observed_count=20,
        densified_count=24,
    ) > _source_patch_feedback_score(
        "source_luma_score",
        patch=flat_patch,
        sample_count=20,
        observed_count=20,
        densified_count=24,
    )
    assert _source_patch_feedback_score(
        "confident_texel_score",
        patch=detailed_patch,
        sample_count=128,
        observed_count=128,
        densified_count=128,
    ) > _source_patch_feedback_score(
        "confident_texel_score",
        patch=detailed_patch,
        sample_count=2,
        observed_count=2,
        densified_count=64,
    )
    assert _source_patch_feedback_score(
        "source_evidence_score",
        patch=detailed_patch,
        sample_count=0,
        observed_count=0,
        densified_count=0,
    ) == 1.0


def test_source_patch_quality_uses_bounded_unique_rgb_for_large_patches() -> None:
    yy, xx = np.indices((512, 512), dtype=np.uint16)
    detailed_patch = np.stack(
        [
            (xx % 256).astype(np.uint8),
            (yy % 256).astype(np.uint8),
            ((xx + yy) % 256).astype(np.uint8),
        ],
        axis=-1,
    )
    flat_patch = np.full_like(detailed_patch, 128)

    assert _bounded_unique_rgb_count(flat_patch, max_samples=4096) == 1
    assert _source_patch_unique_rgb_count(flat_patch) == 1
    assert 16 <= _bounded_unique_rgb_count(detailed_patch, max_samples=4096) <= 4096
    assert 16 <= _source_patch_unique_rgb_count(detailed_patch) <= 4096
    assert _source_patch_quality_score(
        "left_eye",
        patch=detailed_patch,
        sample_count=512,
        foreground_count=1024,
    ) > _source_patch_quality_score(
        "left_eye",
        patch=flat_patch,
        sample_count=512,
        foreground_count=1024,
    )


def test_large_feature_component_masks_use_bounded_downsample() -> None:
    mask = np.zeros((600, 600), dtype=np.bool_)
    mask[120:500, 140:520] = True
    luma = np.ones((600, 600), dtype=np.float64)
    luma[220:360, 260:380] = 0.15

    components = _feature_components(mask, luma_image=luma)

    assert components
    assert max(int(component["stride"]) for component in components) > 1
    selected, metrics = _localized_feature_component_mask(
        mask,
        luma_image=luma,
        expected_center_xy=(320.0, 300.0),
        min_pixels=32,
        max_components=1,
        fallback_mask=np.zeros_like(mask),
    )
    assert metrics["candidate_count"] == int(np.count_nonzero(mask))
    assert int(metrics["selected_count"]) > 0
    assert np.all(selected <= mask)


def test_localized_feature_fallback_stays_inside_nonempty_candidate() -> None:
    candidate = np.zeros((16, 16), dtype=np.bool_)
    candidate[7:9, 7:9] = True
    fallback = np.zeros_like(candidate)
    fallback[4:12, 4:12] = True
    luma = np.ones((16, 16), dtype=np.float64)

    selected, metrics = _localized_feature_component_mask(
        candidate,
        luma_image=luma,
        expected_center_xy=(8.0, 8.0),
        min_pixels=8,
        max_components=1,
        fallback_mask=fallback,
    )

    assert np.array_equal(selected, candidate)
    assert metrics["candidate_count"] == 4
    assert metrics["selected_count"] == 4
    assert metrics["selected_fraction"] == 1.0


def test_large_source_shape_masks_use_bounded_morphology() -> None:
    local_mask = np.zeros((600, 600), dtype=np.bool_)
    local_mask[240:360, 260:380] = True

    shape_mask = _source_shape_mask_from_local_mask(local_mask, sample_key="nose")

    assert shape_mask.shape == local_mask.shape
    assert np.all(shape_mask[local_mask])
    assert np.count_nonzero(shape_mask) > np.count_nonzero(local_mask)
    assert np.count_nonzero(shape_mask) < local_mask.size


def test_detail_preserving_visibility_reinforcement_keeps_source_luma_detail() -> None:
    basecolor = np.full((8, 8, 3), 96, dtype=np.uint8)
    mask = np.zeros((8, 8), dtype=np.bool_)
    mask[2:6, 2:6] = True
    values = np.arange(16, dtype=np.uint8).reshape(4, 4) * 8
    basecolor[2:6, 2:6, 0] = 48 + values
    basecolor[2:6, 2:6, 1] = 42 + values
    basecolor[2:6, 2:6, 2] = 36 + values

    flat = basecolor.copy()
    detailed = basecolor.copy()
    target = np.asarray([180, 170, 160], dtype=np.uint8)

    _paint_feature_visibility_reinforcement(
        flat,
        mask,
        target,
        alpha=0.5,
        preserve_detail=False,
    )
    _paint_feature_visibility_reinforcement(
        detailed,
        mask,
        target,
        alpha=0.5,
        preserve_detail=True,
    )

    source_luma = basecolor[mask].astype(np.float64).mean(axis=1)
    flat_luma = flat[mask].astype(np.float64).mean(axis=1)
    detailed_luma = detailed[mask].astype(np.float64).mean(axis=1)

    assert np.std(detailed_luma) > np.std(flat_luma)
    assert abs(np.std(detailed_luma) - np.std(source_luma)) <= 1.0
    assert np.mean(detailed_luma) > np.mean(source_luma)


def test_observed_iris_detail_gets_reinforcement_floor() -> None:
    assert (
        _eye_source_detail_reinforcement_alpha(
            "left_eye_iris",
            0.0,
            source_patch_luma_std=0.02,
            source_patch_unique_rgb_count=32,
        )
        == 0.18
    )
    assert (
        _eye_source_detail_reinforcement_alpha(
            "right_eye_iris",
            0.60,
            source_patch_luma_std=0.02,
            source_patch_unique_rgb_count=32,
        )
        == 0.60
    )
    assert (
        _eye_source_detail_reinforcement_alpha(
            "left_eye_iris",
            0.0,
            source_patch_luma_std=0.001,
            source_patch_unique_rgb_count=32,
        )
        == 0.0
    )
    assert (
        _eye_source_detail_reinforcement_alpha(
            "left_eye_pupil",
            0.0,
            source_patch_luma_std=0.02,
            source_patch_unique_rgb_count=32,
        )
        == 0.0
    )


def test_side_specific_eye_shape_masks_stay_compact() -> None:
    local_mask = np.zeros((16, 16), dtype=np.bool_)
    local_mask[4:7, 5:8] = True
    local_mask[9:11, 10:12] = True

    broad_eye = _source_shape_mask_from_local_mask(local_mask, sample_key="left_eye")
    compact_iris = _source_shape_mask_from_local_mask(local_mask, sample_key="left_iris")
    compact_pupil = _source_shape_mask_from_local_mask(local_mask, sample_key="left_pupil")

    assert np.count_nonzero(compact_iris) < np.count_nonzero(broad_eye)
    assert np.count_nonzero(compact_pupil) < np.count_nonzero(broad_eye)
    assert not np.any(compact_iris[9:11, 10:12])
    assert not np.any(compact_pupil[9:11, 10:12])
    assert np.all(compact_pupil[4:7, 5:8])


def test_broad_eye_shape_mask_compacts_diagonal_artifact() -> None:
    local_mask = np.zeros((48, 32), dtype=np.bool_)
    for offset in range(30):
        y = 7 + offset
        x = 8 + offset // 2
        local_mask[y : y + 3, x : x + 5] = True

    compact_eye = _source_shape_mask_from_local_mask(local_mask, sample_key="left_eye")
    iris_mask = _source_shape_mask_from_local_mask(local_mask, sample_key="left_iris")

    assert np.count_nonzero(compact_eye) < np.count_nonzero(local_mask)
    assert np.count_nonzero(compact_eye) > np.count_nonzero(iris_mask)
    assert not np.any(compact_eye[7:10, 8:13])
    assert not np.any(compact_eye[35:40, 21:26])
    assert np.any(compact_eye[20:27, 14:20])


def test_broad_eye_shape_mask_keeps_nonelongated_support_broad() -> None:
    local_mask = np.zeros((20, 24), dtype=np.bool_)
    local_mask[7:13, 8:16] = True

    broad_eye = _source_shape_mask_from_local_mask(local_mask, sample_key="right_eye")
    compact_iris = _source_shape_mask_from_local_mask(local_mask, sample_key="right_iris")

    assert np.count_nonzero(broad_eye) > np.count_nonzero(local_mask)
    assert np.count_nonzero(broad_eye) > np.count_nonzero(compact_iris)


def test_side_specific_iris_destination_footprint_is_tightly_layered() -> None:
    patch = np.zeros((32, 32, 3), dtype=np.uint8)

    unaligned = _source_patch_footprint(
        "left_iris",
        nominal_radius_uv=(0.032, 0.022),
        patch=patch,
        sample_count=128,
        used_source_patch=True,
        landmark_confidence=0.0,
    )
    aligned = _source_patch_footprint(
        "left_iris",
        nominal_radius_uv=(0.032, 0.022),
        patch=patch,
        sample_count=128,
        used_source_patch=True,
        landmark_confidence=1.0,
    )

    assert aligned.landmark_destination_scale == 0.50
    assert aligned.area_scale < unaligned.area_scale
    assert aligned.radius_uv[0] < unaligned.radius_uv[0]
    assert aligned.radius_uv[1] < unaligned.radius_uv[1]


def test_side_specific_iris_destination_uses_accepted_core_count() -> None:
    patch = np.zeros((37, 265, 3), dtype=np.uint8)
    candidate_count = 680
    accepted_core_count = 227

    candidate_driven = _source_patch_footprint(
        "right_iris",
        nominal_radius_uv=(0.032, 0.022),
        patch=patch,
        sample_count=candidate_count,
        used_source_patch=True,
        landmark_confidence=1.0,
    )
    core_driven = _source_patch_footprint(
        "right_iris",
        nominal_radius_uv=(0.032, 0.022),
        patch=patch,
        sample_count=_source_patch_destination_sample_count(
            "right_iris",
            sample_count=candidate_count,
            source_mask_observed_texel_count=accepted_core_count,
        ),
        used_source_patch=True,
        landmark_confidence=1.0,
    )

    assert _source_patch_destination_sample_count(
        "right_iris",
        sample_count=candidate_count,
        source_mask_observed_texel_count=accepted_core_count,
    ) == accepted_core_count
    assert _source_patch_destination_sample_count(
        "right_eye",
        sample_count=candidate_count,
        source_mask_observed_texel_count=accepted_core_count,
    ) == candidate_count
    assert core_driven.area_scale < candidate_driven.area_scale
    assert core_driven.radius_uv[0] < candidate_driven.radius_uv[0]
    assert core_driven.radius_uv[1] < candidate_driven.radius_uv[1]


def test_iris_source_luma_core_rejects_bright_annulus() -> None:
    crop = np.full((12, 12, 3), 0.72, dtype=np.float32)
    local_mask = np.zeros((12, 12), dtype=np.bool_)
    local_mask[2:10, 2:10] = True
    crop[4:8, 4:8, :] = 0.08

    iris_core = _source_observed_mask_from_local_mask(
        crop,
        local_mask,
        sample_key="left_iris",
    )
    broad_eye = _source_observed_mask_from_local_mask(
        crop,
        local_mask,
        sample_key="left_eye",
    )
    direct_core = _iris_source_luma_core_mask(crop, local_mask)

    assert np.array_equal(iris_core, direct_core)
    assert np.array_equal(broad_eye, local_mask)
    assert np.count_nonzero(iris_core) < np.count_nonzero(local_mask)
    assert np.all(iris_core[4:8, 4:8])
    assert not np.any(iris_core[2, 2:10])


def test_iris_patch_paint_clamps_edge_support_to_core() -> None:
    basecolor = np.zeros((17, 17, 3), dtype=np.uint8)
    mask = np.zeros((17, 17), dtype=np.bool_)
    mask[2:15, 2:15] = True
    patch = np.full((13, 13, 3), 255, dtype=np.uint8)
    source_shape_mask = np.full((13, 13), 255, dtype=np.uint8)

    stats = _paint_patch_into_mask(
        basecolor,
        mask,
        patch,
        sample_key="right_iris",
        source_shape_mask=source_shape_mask,
        fusion_weight=1.0,
        feather_width=0.95,
        feather_x_scale=1.02,
        feather_y_scale=0.90,
    )

    assert stats.core_texel_count > 0
    assert stats.feathered_edge_texel_count > 0
    assert stats.edge_clamp_removed_texel_count > 0
    assert stats.edge_clamp_max_ratio == 0.75
    assert stats.feathered_edge_texel_count <= int(
        np.ceil(stats.core_texel_count * stats.edge_clamp_max_ratio)
    )


def test_iris_patch_paint_preserves_broader_edge_for_large_core() -> None:
    basecolor = np.zeros((65, 65, 3), dtype=np.uint8)
    mask = np.zeros((65, 65), dtype=np.bool_)
    mask[8:57, 8:57] = True
    patch = np.full((49, 49, 3), 255, dtype=np.uint8)
    source_shape_mask = np.full((49, 49), 255, dtype=np.uint8)

    stats = _paint_patch_into_mask(
        basecolor,
        mask,
        patch,
        sample_key="right_iris",
        source_shape_mask=source_shape_mask,
        fusion_weight=1.0,
        feather_width=0.05,
        feather_x_scale=1.02,
        feather_y_scale=0.90,
    )

    assert stats.core_texel_count > 32
    assert stats.edge_clamp_max_ratio == 1.25
    assert stats.feathered_edge_texel_count <= int(
        np.ceil(stats.core_texel_count * stats.edge_clamp_max_ratio)
    )


def test_iris_patch_paint_uses_env_clamp_ratio(monkeypatch) -> None:
    monkeypatch.setenv("OPENCAT3D_IRIS_EDGE_CORE_RATIO", "0.50")
    basecolor = np.zeros((17, 17, 3), dtype=np.uint8)
    mask = np.zeros((17, 17), dtype=np.bool_)
    mask[2:15, 2:15] = True
    patch = np.full((13, 13, 3), 255, dtype=np.uint8)
    source_shape_mask = np.full((13, 13), 255, dtype=np.uint8)

    stats = _paint_patch_into_mask(
        basecolor,
        mask,
        patch,
        sample_key="right_iris",
        source_shape_mask=source_shape_mask,
        fusion_weight=1.0,
        feather_width=0.95,
        feather_x_scale=1.02,
        feather_y_scale=0.90,
    )

    assert stats.core_texel_count > 0
    assert stats.edge_clamp_max_ratio == 0.50
    assert stats.feathered_edge_texel_count <= int(
        np.ceil(stats.core_texel_count * 0.50)
    )
