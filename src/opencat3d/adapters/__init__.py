"""External adapter interfaces."""

from opencat3d.adapters.dense_evidence import (
    DENSE_EVIDENCE_MANIFEST_FILENAME,
    DenseEvidencePayload,
    DenseEvidenceProvider,
    DenseEvidenceRequest,
    LocalFileDenseEvidenceProvider,
    UnavailableDenseEvidenceProvider,
    unavailable_dense_evidence_provider,
)

__all__ = [
    "DENSE_EVIDENCE_MANIFEST_FILENAME",
    "DenseEvidenceProvider",
    "DenseEvidencePayload",
    "DenseEvidenceRequest",
    "LocalFileDenseEvidenceProvider",
    "UnavailableDenseEvidenceProvider",
    "unavailable_dense_evidence_provider",
]
