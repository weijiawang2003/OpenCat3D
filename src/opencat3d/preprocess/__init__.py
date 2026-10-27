"""Preprocessing modules."""

from opencat3d.preprocess.frame_selection import (
    RawFrameSelectionResult,
    RawTextureManifestDedupResult,
    deduplicate_raw_texture_manifest_by_image_hash,
    select_raw_observation_frames,
)
from opencat3d.preprocess.keypoints import (
    KeypointBootstrapFrameResult,
    KeypointValidation,
    RawKeypointBootstrapResult,
    SyntheticKeypointBootstrapResult,
    bootstrap_raw_keypoints_and_pose,
    bootstrap_synthetic_keypoints_and_pose,
    validate_keypoints_xyc,
)
from opencat3d.preprocess.segmentation import (
    RawSegmentationBackend,
    RawSegmentationResult,
    SegmentationFrameResult,
    SyntheticSegmentationResult,
    bbox_from_mask,
    load_image_srgb,
    segment_raw_foreground,
    segment_raw_image_observations,
    segment_synthetic_rgb,
    segment_synthetic_rgb_observations,
)

__all__ = [
    "KeypointBootstrapFrameResult",
    "KeypointValidation",
    "RawFrameSelectionResult",
    "RawTextureManifestDedupResult",
    "RawKeypointBootstrapResult",
    "RawSegmentationResult",
    "RawSegmentationBackend",
    "SegmentationFrameResult",
    "SyntheticKeypointBootstrapResult",
    "SyntheticSegmentationResult",
    "bbox_from_mask",
    "bootstrap_raw_keypoints_and_pose",
    "bootstrap_synthetic_keypoints_and_pose",
    "deduplicate_raw_texture_manifest_by_image_hash",
    "load_image_srgb",
    "segment_raw_foreground",
    "segment_raw_image_observations",
    "segment_synthetic_rgb",
    "segment_synthetic_rgb_observations",
    "select_raw_observation_frames",
    "validate_keypoints_xyc",
]
