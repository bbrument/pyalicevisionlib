#!/usr/bin/env python3
"""
Convert AliceVision SfMData JSON to IDR cameras.npz.

AliceVision SfMData format:
- Rotation: cam2world (3x3, row-major, 9 floats)
- Center: camera center in world coordinates (3 floats)
- Intrinsics: focalLength (mm), sensorWidth (mm), principalPoint (offset from center, px)

IDR format (cameras.npz):
- world_mat_{i}: 4x4 projection matrix P = K @ [R|t] (bottom row [0,0,0,1])
- scale_mat_{i}: 4x4 normalization transform (normalized space -> world space)

Scale matrix modes:
- identity: scale_mat = I_4x4 (default)
- masks: shape-from-silhouette from mask images
- mesh: bounding sphere from mesh/PCD vertices
- file: load existing scale_mats.npz

Usage:
    pyav-sfm2idr sfmdata.json output_dir
    pyav-sfm2idr sfmdata.json output_dir --scale-mode masks --masks-folder /path/to/masks
    pyav-sfm2idr sfmdata.json output_dir --scale-mode mesh --geometry mesh.ply
    pyav-sfm2idr sfmdata.json output_dir --scale-mode file --scale-mats scale_mats.npz
"""

import argparse
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ..sfmdata import SfMDataWrapper, WORLD_CORRECTION


def av_rotation_to_idr(av_rotation: np.ndarray) -> np.ndarray:
    """Convert AV cam2world rotation to IDR world-to-camera R_w2c.

    WORLD_CORRECTION is self-inverse (diagonal with 1, -1, -1):
        R_cam2world_orig = WORLD_CORRECTION @ av_rotation
        R_w2c = R_cam2world_orig.T

    Args:
        av_rotation: 3x3 cam2world rotation from AliceVision (already world-corrected)

    Returns:
        3x3 world-to-camera rotation matrix
    """
    R_cam2world_orig = WORLD_CORRECTION @ av_rotation
    R_w2c = R_cam2world_orig.T
    return R_w2c


def av_center_to_idr(av_center: np.ndarray) -> np.ndarray:
    """Convert AV camera center to IDR original coordinates.

    WORLD_CORRECTION is self-inverse:
        center_orig = WORLD_CORRECTION @ av_center

    Args:
        av_center: Camera center from AliceVision (already world-corrected)

    Returns:
        Camera center in original IDR coordinates (3,)
    """
    return WORLD_CORRECTION @ av_center


def build_world_mat(K: np.ndarray, R_w2c: np.ndarray, center: np.ndarray) -> np.ndarray:
    """Build IDR world_mat (4x4) from K, R_w2c, and camera center.

    P = K @ [R_w2c | t] where t = -R_w2c @ center
    Extended to 4x4 with bottom row [0, 0, 0, 1].

    Args:
        K: 3x3 intrinsic matrix
        R_w2c: 3x3 world-to-camera rotation
        center: Camera center in world coordinates (3,)

    Returns:
        4x4 projection matrix (float64 — caller can cast to float32 if needed)
    """
    t = -R_w2c @ center
    Rt = np.hstack([R_w2c, t.reshape(3, 1)])  # 3x4
    P_3x4 = K @ Rt
    P = np.vstack([P_3x4, [0, 0, 0, 1]])
    return P.astype(np.float64)


def build_K_from_intrinsics(
    focal_mm: float, sensor_width: float, width: int, height: int,
    pp_offset: np.ndarray, pixel_ratio: float = 1.0
) -> np.ndarray:
    """Build 3x3 intrinsic matrix from AliceVision intrinsics.

    Args:
        focal_mm: Focal length in mm
        sensor_width: Sensor width in mm
        width: Image width in pixels
        height: Image height in pixels
        pp_offset: Principal point offset from image center [cx_offset, cy_offset]
        pixel_ratio: fy/fx ratio (default 1.0 for square pixels)

    Returns:
        3x3 intrinsic matrix
    """
    fx = focal_mm * width / sensor_width
    fy = fx * pixel_ratio
    cx = width / 2.0 + pp_offset[0]
    cy = height / 2.0 + pp_offset[1]
    return np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1],
    ], dtype=np.float64)


def compute_scale_mat_identity() -> np.ndarray:
    """Return identity scale matrix (4x4 float32)."""
    return np.eye(4, dtype=np.float32)


def compute_scale_mat_from_geometry(geometry_path: str) -> np.ndarray:
    """Compute scale matrix from mesh/PCD bounding sphere.

    scale_mat maps normalized space (unit sphere) to world space:
        point_world = scale_mat @ point_normalized

    Args:
        geometry_path: Path to mesh or point cloud file (.ply, .obj, etc.)

    Returns:
        4x4 scale matrix (float32)
    """
    import trimesh

    loaded = trimesh.load(geometry_path, process=False)

    # Handle Scene (multiple meshes) by concatenating vertices
    if isinstance(loaded, trimesh.Scene):
        all_verts = []
        for geom in loaded.geometry.values():
            if hasattr(geom, "vertices"):
                all_verts.append(np.array(geom.vertices))
        if not all_verts:
            raise ValueError(f"No geometry with vertices found in {geometry_path}")
        vertices = np.vstack(all_verts)
    elif hasattr(loaded, "vertices"):
        vertices = np.array(loaded.vertices)
    else:
        raise ValueError(f"Cannot extract vertices from {geometry_path}")

    centroid = np.mean(vertices, axis=0)
    radius = np.max(np.linalg.norm(vertices - centroid, axis=1))

    S = np.eye(4, dtype=np.float32)
    S[:3, :3] *= radius
    S[:3, 3] = centroid.astype(np.float32)
    return S


def compute_scale_mat_from_masks(
    masks_folder: str,
    world_mats: List[np.ndarray],
    width: int,
    height: int,
) -> np.ndarray:
    """Compute scale matrix via shape-from-silhouette.

    Inspired by IDR's preprocess_cameras.py:
    - Create voxel grid (100x100x100) spanning camera bounding volume
    - Project each voxel to all views, check if inside mask
    - Retain voxels visible in >= max(1, n_views // 2) views
    - Compute centroid and scale from bounding sphere radius

    Args:
        masks_folder: Path to mask images (000.png, 001.png, ...)
        world_mats: List of 4x4 world_mat projection matrices
        width: Image width
        height: Image height

    Returns:
        4x4 scale matrix (float32)
    """
    from ..image import load_mask

    masks_path = Path(masks_folder)
    # Glob all mask extensions then sort combined list
    _mask_exts = ["png", "PNG", "jpg", "JPG", "jpeg", "JPEG", "tif", "TIF", "exr", "EXR"]
    mask_files = sorted(set(
        p for ext in _mask_exts for p in masks_path.glob(f"*.{ext}")
    ))
    if not mask_files:
        print("Warning: No mask files found, falling back to identity scale_mat")
        return compute_scale_mat_identity()

    n_views = len(world_mats)

    if len(mask_files) < n_views:
        print(
            f"Warning: Found {len(mask_files)} masks but {n_views} views. "
            f"Using {len(mask_files)} masks."
        )

    # Load masks
    masks = []
    for mf in mask_files[:n_views]:
        masks.append(load_mask(str(mf)))

    # Extract camera centers from world_mats via null space of P
    centers = []
    for P in world_mats:
        _, _, Vt = np.linalg.svd(P[:3, :4])
        C_h = Vt[-1]
        C = C_h[:3] / C_h[3]
        centers.append(C)
    centers = np.array(centers)

    # Compute bounding volume from camera centers
    extent = centers.max(axis=0) - centers.min(axis=0)
    bbox_min = centers.min(axis=0) - 2 * extent
    bbox_max = centers.max(axis=0) + 2 * extent

    # Create voxel grid
    grid_res = 100
    x = np.linspace(bbox_min[0], bbox_max[0], grid_res)
    y = np.linspace(bbox_min[1], bbox_max[1], grid_res)
    z = np.linspace(bbox_min[2], bbox_max[2], grid_res)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    voxels = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)  # (N, 3)
    n_voxels = voxels.shape[0]

    # Project voxels to each view and check visibility
    vote_count = np.zeros(n_voxels, dtype=np.int32)
    voxels_h = np.hstack([voxels, np.ones((n_voxels, 1))])  # (N, 4)

    for i in range(min(n_views, len(masks))):
        P = world_mats[i][:3, :4]
        projected = (P @ voxels_h.T).T  # (N, 3)

        depth = projected[:, 2]
        valid = depth > 0

        u = np.full(n_voxels, -1.0)
        v = np.full(n_voxels, -1.0)
        u[valid] = projected[valid, 0] / depth[valid]
        v[valid] = projected[valid, 1] / depth[valid]

        # Check if pixel is inside image and inside mask
        u_int = np.round(u).astype(np.int32)
        v_int = np.round(v).astype(np.int32)

        in_image = valid & (u_int >= 0) & (u_int < width) & (v_int >= 0) & (v_int < height)

        mask = masks[i]
        in_mask = np.zeros(n_voxels, dtype=bool)
        in_mask[in_image] = mask[v_int[in_image], u_int[in_image]]

        vote_count[in_mask] += 1

    # Retain voxels visible in at least half the views (min 1)
    threshold = max(1, n_views // 2)
    retained = voxels[vote_count >= threshold]

    if len(retained) == 0:
        print("Warning: No voxels retained by shape-from-silhouette, falling back to identity")
        return compute_scale_mat_identity()

    centroid = np.mean(retained, axis=0)
    # Use bounding sphere radius from centroid (more robust than flattened std)
    scale = 3.0 * np.max(np.std(retained, axis=0))

    S = np.eye(4, dtype=np.float32)
    S[:3, :3] *= scale
    S[:3, 3] = centroid.astype(np.float32)
    return S


def convert_sfmdata_to_idr(
    sfmdata_path: str,
    output_folder: str,
    scale_mode: str = "identity",
    masks_folder: Optional[str] = None,
    geometry_path: Optional[str] = None,
    scale_mats_path: Optional[str] = None,
) -> Dict:
    """Convert AliceVision SfMData to IDR cameras.npz.

    Args:
        sfmdata_path: Path to input SfMData file
        output_folder: Output IDR directory
        scale_mode: Scale matrix mode: "identity", "masks", "mesh", "file"
        masks_folder: Path to masks (for scale_mode=masks and mask copying)
        geometry_path: Path to mesh/PCD (for scale_mode=mesh)
        scale_mats_path: Path to existing scale_mats.npz (for scale_mode=file)

    Returns:
        Dict with conversion statistics
    """
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load SfMData
    sfm = SfMDataWrapper.load(sfmdata_path)
    sfm_dict = sfm.as_dict()
    print(f"Loaded SfMData: {sfmdata_path}")

    # Build lookup dicts
    intrinsics_dict = {}
    for intr in sfm_dict.get("intrinsics", []):
        intrinsics_dict[intr["intrinsicId"]] = intr

    poses_dict = {}
    for pose in sfm_dict.get("poses", []):
        poses_dict[pose["poseId"]] = pose["pose"]["transform"]

    views = sfm_dict.get("views", [])
    # Sort by frameId (preserves IDR camera index ordering), then numeric viewId
    views = sorted(
        views,
        key=lambda v: (int(v.get("frameId", 0)), int(v.get("viewId", 0)))
    )

    # Distortion warning (once)
    for intr in intrinsics_dict.values():
        dist_type = intr.get("distortionType", "none")
        if dist_type != "none":
            print(
                f"Warning: SfMData contains distortion parameters ({dist_type}) "
                f"which will be discarded. IDR format assumes undistorted images."
            )
            break

    world_mats = []
    first_width = None
    first_height = None

    for view in views:
        intr_id = view["intrinsicId"]
        pose_id = view["poseId"]

        if intr_id not in intrinsics_dict or pose_id not in poses_dict:
            continue

        intr = intrinsics_dict[intr_id]
        transform = poses_dict[pose_id]

        width = int(intr["width"])
        height = int(intr["height"])
        focal_mm = float(intr["focalLength"])
        sensor_width = float(intr.get("sensorWidth", 36.0))
        pixel_ratio = float(intr.get("pixelRatio", 1.0))

        if first_width is None:
            first_width = width
            first_height = height

        pp = intr.get("principalPoint", ["0", "0"])
        pp_offset = np.array([float(pp[0]), float(pp[1])])

        # Parse rotation and center
        rotation_flat = [float(r) for r in transform["rotation"]]
        R_cam2world_av = np.array(rotation_flat).reshape(3, 3)
        center_av = np.array([float(c) for c in transform["center"]])

        # Convert to IDR coordinates
        R_w2c = av_rotation_to_idr(R_cam2world_av)
        center_orig = av_center_to_idr(center_av)

        # Build K and world_mat
        K = build_K_from_intrinsics(
            focal_mm, sensor_width, width, height, pp_offset, pixel_ratio
        )
        P = build_world_mat(K, R_w2c, center_orig)

        world_mats.append(P)

    n_views = len(world_mats)

    if n_views == 0:
        raise ValueError("No valid views found in SfMData (all views missing pose or intrinsics)")

    print(f"Converted {n_views} views")

    # Compute scale matrices
    if scale_mode == "identity":
        scale_mat = compute_scale_mat_identity()
        scale_mats = [scale_mat] * n_views
    elif scale_mode == "mesh":
        if not geometry_path:
            raise ValueError("--geometry required for scale_mode=mesh")
        scale_mat = compute_scale_mat_from_geometry(geometry_path)
        scale_mats = [scale_mat] * n_views
    elif scale_mode == "masks":
        if not masks_folder:
            raise ValueError("--masks-folder required for scale_mode=masks")
        scale_mat = compute_scale_mat_from_masks(
            masks_folder, world_mats, first_width, first_height
        )
        scale_mats = [scale_mat] * n_views
    elif scale_mode == "file":
        if not scale_mats_path:
            raise ValueError("--scale-mats required for scale_mode=file")
        scale_data = np.load(scale_mats_path)
        scale_mats = [
            scale_data[f"scale_mat_{i}"].astype(np.float32) for i in range(n_views)
        ]
    else:
        raise ValueError(f"Unknown scale_mode: {scale_mode}")

    # Save cameras.npz (float32 for IDR compatibility)
    npz_data = {}
    for i in range(n_views):
        npz_data[f"world_mat_{i}"] = world_mats[i].astype(np.float32)
        npz_data[f"scale_mat_{i}"] = scale_mats[i]

    np.savez(str(output_path / "cameras.npz"), **npz_data)
    print(f"Wrote cameras.npz: {n_views} views, scale_mode={scale_mode}")

    # Copy masks if provided
    masks_copied = 0
    if masks_folder:
        mask_output = output_path / "mask"
        mask_output.mkdir(parents=True, exist_ok=True)

        mask_src = Path(masks_folder)
        _copy_exts = ["png", "PNG", "jpg", "JPG", "jpeg", "JPEG", "tif", "TIF", "exr", "EXR"]
        mask_files = sorted(set(
            p for ext in _copy_exts for p in mask_src.glob(f"*.{ext}")
        ))

        for i, mf in enumerate(mask_files[:n_views]):
            # Preserve original extension to avoid format mismatch
            dst = mask_output / f"{i:03d}{mf.suffix}"
            shutil.copy2(str(mf), str(dst))
            masks_copied += 1

        print(f"Copied {masks_copied} masks to {mask_output}")

    return {
        "n_views": n_views,
        "scale_mode": scale_mode,
        "masks_copied": masks_copied,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Convert AliceVision SfMData to IDR cameras.npz",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  pyav-sfm2idr sfmdata.json output_dir
  pyav-sfm2idr sfmdata.json output_dir --scale-mode masks --masks-folder /path/to/masks
  pyav-sfm2idr sfmdata.json output_dir --scale-mode mesh --geometry mesh.ply
  pyav-sfm2idr sfmdata.json output_dir --scale-mode file --scale-mats scale_mats.npz
        """,
    )
    parser.add_argument("sfmdata", help="Input SfMData file")
    parser.add_argument("output_folder", help="Output IDR directory")
    parser.add_argument(
        "--scale-mode",
        choices=["identity", "masks", "mesh", "file"],
        default="identity",
        help="Scale matrix computation mode (default: identity)",
    )
    parser.add_argument(
        "--masks-folder",
        help="Path to masks (for scale_mode=masks and for copying to output)",
    )
    parser.add_argument(
        "--geometry",
        help="Path to mesh/PCD (for scale_mode=mesh)",
    )
    parser.add_argument(
        "--scale-mats",
        help="Path to existing scale_mats.npz (for scale_mode=file)",
    )

    args = parser.parse_args()

    convert_sfmdata_to_idr(
        sfmdata_path=args.sfmdata,
        output_folder=args.output_folder,
        scale_mode=args.scale_mode,
        masks_folder=args.masks_folder,
        geometry_path=args.geometry,
        scale_mats_path=args.scale_mats,
    )


if __name__ == "__main__":
    main()
