"""
pyalicevisionlib - Python utilities for AliceVision/Meshroom.

This library provides:
- Camera handling (intrinsics, extrinsics, projection)
- SfMData loading (via pyalicevision or JSON fallback)
- Mesh loading and processing
- Mesh evaluation (Chamfer distance, precision/recall)
- Visibility-based mesh cleanup
- Visualization tools

Coordinate Conventions:
- Rotation: cam2world (camera axes in world coordinates)
- Center: camera position in world coordinates
- Principal point: offset from image center in pixels
"""

from .camera import Camera
from .sfmdata import (
    # Main classes
    SfMDataWrapper,
    # New unified API
    load_sfmdata,
    save_sfmdata,
    # Legacy API (still supported)
    load_cameras_from_sfmdata,
    get_camera_centers,
    # Feature detection
    HAS_PYALICEVISION,
    # Constants
    WORLD_CORRECTION,
    # Legacy JSON-only functions (deprecated but kept for compatibility)
    load_sfmdata_json,
    save_sfmdata_json,
    get_viewid_by_image_name,
    get_viewid_to_image_path,
)
from .mesh import (
    load_mesh,
    sample_mesh_points,
)
from .utils import (
    load_mesh_arrays,
    save_pointcloud,
    upsample_mesh_to_pointcloud,
    downsample_pointcloud,
    compute_nearest_neighbor_distances,
    remove_mesh_vertices_by_mask,
    remove_mesh_vertices_below_threshold,
    load_transform,
    decompose_transform,
)
from .evaluation import (
    ChamferResult,
    PrecisionRecallResult,
    evaluate_mesh,
    compute_chamfer_distance,
    compute_precision_recall,
    mesh_to_pointcloud,
    cleanup_mesh_visibility,
    cleanup_mesh_with_masks,
    run_evaluation,
)
from .image import (
    load_image,
    load_gray,
    load_mask,
    load_alpha_as_mask,
    save_image,
    get_image_dimensions,
    get_available_backends,
)

__version__ = "0.1.0"

__all__ = [
    # Version
    "__version__",
    # Camera
    "Camera",
    # SfMData - Main classes
    "SfMDataWrapper",
    # SfMData - New unified API
    "load_sfmdata",
    "save_sfmdata",
    # SfMData - Legacy API
    "load_cameras_from_sfmdata",
    "get_camera_centers",
    "HAS_PYALICEVISION",
    "WORLD_CORRECTION",
    # SfMData - Legacy JSON functions (deprecated)
    "load_sfmdata_json",
    "save_sfmdata_json",
    "get_viewid_by_image_name",
    "get_viewid_to_image_path",
    # Mesh
    "load_mesh",
    "sample_mesh_points",
    # Utils
    "load_mesh_arrays",
    "save_pointcloud",
    "upsample_mesh_to_pointcloud",
    "downsample_pointcloud",
    "compute_nearest_neighbor_distances",
    "remove_mesh_vertices_by_mask",
    "remove_mesh_vertices_below_threshold",
    "load_transform",
    "decompose_transform",
    # Image I/O
    "load_image",
    "load_gray",
    "load_mask",
    "load_alpha_as_mask",
    "save_image",
    "get_image_dimensions",
    "get_available_backends",
    # Evaluation
    "ChamferResult",
    "PrecisionRecallResult",
    "evaluate_mesh",
    "compute_chamfer_distance",
    "compute_precision_recall",
    "mesh_to_pointcloud",
    "cleanup_mesh_visibility",
    "cleanup_mesh_with_masks",
    "run_evaluation",
]
