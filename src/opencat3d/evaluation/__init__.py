"""Evaluation modules."""

from opencat3d.evaluation.acceptance import (
    SYNTHETIC_ACCEPTANCE_THRESHOLDS,
    SyntheticAcceptanceAuditResult,
    evaluate_synthetic_acceptance,
)
from opencat3d.evaluation.compliance import (
    ObjectiveComplianceResult,
    write_objective_compliance_audit,
    write_raw_objective_compliance_audit,
)
from opencat3d.evaluation.coverage import (
    PROVENANCE_LABELS,
    MeshCoverageResult,
    evaluate_mesh_coverage,
)
from opencat3d.evaluation.evidence_gap import (
    CAPTURE_TARGETS,
    SourceFrameAcquisitionGapResult,
    write_source_frame_acquisition_gap_report,
)
from opencat3d.evaluation.face_texture_bake_comparison import (
    FaceTextureBakeComparisonResult,
    evaluate_face_texture_bake_comparison_report,
)
from opencat3d.evaluation.face_texture_targets import (
    FaceTextureTargetEvaluationResult,
    evaluate_face_texture_targets_report,
)
from opencat3d.evaluation.face_uv_density import (
    REQUIRED_FACE_UV_REGIONS,
    FaceUvDensityEvaluationResult,
    evaluate_face_uv_density_report,
)
from opencat3d.evaluation.frame_set_supervisor import (
    BodyFitAlignmentGateResult,
    FrameSetSupervisorResult,
    TextureOnlyPromotionGuardResult,
    write_body_fit_alignment_gate_report,
    write_frame_set_candidate_report,
    write_texture_only_promotion_guard,
)
from opencat3d.evaluation.heldout import (
    SYNTHETIC_HELDOUT_THRESHOLDS,
    HeldoutFrameMetric,
    SyntheticHeldoutEvaluationResult,
    evaluate_synthetic_heldout_views,
)
from opencat3d.evaluation.projection import (
    JOINT_PROJECTION_THRESHOLDS,
    JointProjectionEvaluationResult,
    JointProjectionFrameMetric,
    evaluate_joint_projection_report,
)
from opencat3d.evaluation.raw import (
    RAW_METRIC_THRESHOLDS,
    RawEvaluationResult,
    evaluate_raw_reconstruction,
)
from opencat3d.evaluation.residuals import (
    RESIDUAL_LEGEND,
    SILHOUETTE_RESIDUAL_THRESHOLDS,
    SilhouetteResidualEvaluationResult,
    SilhouetteResidualFrameMetric,
    evaluate_silhouette_residual_maps,
)
from opencat3d.evaluation.supervisor import (
    GuardedCandidateComparisonResult,
    write_guarded_candidate_comparison_report,
)
from opencat3d.evaluation.synthetic import (
    SYNTHETIC_METRIC_THRESHOLDS,
    SyntheticEvaluationResult,
    evaluate_synthetic_reconstruction,
)
from opencat3d.evaluation.texture_probe_regression import (
    TextureProbeRegressionAuditResult,
    write_texture_probe_regression_audit,
)
from opencat3d.evaluation.validation import (
    ArtifactValidationResult,
    write_quality_validation_report,
)

__all__ = [
    "SYNTHETIC_ACCEPTANCE_THRESHOLDS",
    "SYNTHETIC_HELDOUT_THRESHOLDS",
    "SYNTHETIC_METRIC_THRESHOLDS",
    "JOINT_PROJECTION_THRESHOLDS",
    "RAW_METRIC_THRESHOLDS",
    "CAPTURE_TARGETS",
    "PROVENANCE_LABELS",
    "REQUIRED_FACE_UV_REGIONS",
    "RESIDUAL_LEGEND",
    "SILHOUETTE_RESIDUAL_THRESHOLDS",
    "HeldoutFrameMetric",
    "ArtifactValidationResult",
    "FaceTextureBakeComparisonResult",
    "FaceTextureTargetEvaluationResult",
    "FaceUvDensityEvaluationResult",
    "BodyFitAlignmentGateResult",
    "FrameSetSupervisorResult",
    "TextureOnlyPromotionGuardResult",
    "JointProjectionEvaluationResult",
    "JointProjectionFrameMetric",
    "MeshCoverageResult",
    "ObjectiveComplianceResult",
    "RawEvaluationResult",
    "SourceFrameAcquisitionGapResult",
    "SilhouetteResidualEvaluationResult",
    "SilhouetteResidualFrameMetric",
    "SyntheticAcceptanceAuditResult",
    "SyntheticHeldoutEvaluationResult",
    "SyntheticEvaluationResult",
    "GuardedCandidateComparisonResult",
    "TextureProbeRegressionAuditResult",
    "evaluate_synthetic_acceptance",
    "write_objective_compliance_audit",
    "write_raw_objective_compliance_audit",
    "evaluate_synthetic_heldout_views",
    "evaluate_joint_projection_report",
    "evaluate_mesh_coverage",
    "write_source_frame_acquisition_gap_report",
    "evaluate_face_texture_bake_comparison_report",
    "evaluate_face_texture_targets_report",
    "evaluate_face_uv_density_report",
    "write_body_fit_alignment_gate_report",
    "write_frame_set_candidate_report",
    "write_texture_only_promotion_guard",
    "write_guarded_candidate_comparison_report",
    "write_texture_probe_regression_audit",
    "evaluate_raw_reconstruction",
    "evaluate_silhouette_residual_maps",
    "evaluate_synthetic_reconstruction",
    "write_quality_validation_report",
]
