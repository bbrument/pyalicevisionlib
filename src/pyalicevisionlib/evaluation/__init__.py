"""
Evaluation subpackage for mesh quality assessment.

Provides:
- Chamfer distance computation
- Precision/recall metrics
- Visibility-based mesh cleanup
- Full evaluation pipeline
"""

from .chamfer import (
    ChamferResult,
    PrecisionRecallResult,
    compute_chamfer_distance,
    compute_precision_recall,
    mesh_to_pointcloud,
    evaluate_mesh,
)
from .cleanup import (
    cleanup_mesh_visibility,
    cleanup_mesh_with_masks,
)
from .pipeline import run_evaluation

__all__ = [
    "ChamferResult",
    "PrecisionRecallResult",
    "compute_chamfer_distance",
    "compute_precision_recall",
    "mesh_to_pointcloud",
    "evaluate_mesh",
    "cleanup_mesh_visibility",
    "cleanup_mesh_with_masks",
    "run_evaluation",
]
