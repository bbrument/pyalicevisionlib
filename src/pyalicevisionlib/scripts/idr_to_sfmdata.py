#!/usr/bin/env python3
"""
Convert IDR cameras.npz to AliceVision SfMData JSON.

IDR format (cameras.npz):
- world_mat_{i}: 4x4 projection matrix P = K @ [R|t] (bottom row [0,0,0,1])
  Projects world-space 3D points to 2D: d * [u, v, 1] = P[:3,:4] @ [X, Y, Z, 1]
- scale_mat_{i}: 4x4 normalization transform (normalized space -> world space)
  Format: diag(s, s, s, 1) with centroid [cx, cy, cz] in column 3
- Optional: camera_mat_{i} / camera_mat_inv_{i} for numerically stable decomposition

AliceVision SfMData format:
- Rotation: cam2world (3x3, row-major, 9 floats)
- Center: camera center in world coordinates (3 floats)
- Intrinsics: focalLength (mm), sensorWidth (mm), principalPoint (offset from center, px)

Note: When no --images-folder is provided, image dimensions are inferred from K
(cx ~ width/2, cy ~ height/2). This is approximate if the principal point has a
significant offset. Prefer providing --images-folder for accurate dimensions.

Usage:
    pyav-idr2sfm cameras.npz output.json
    pyav-idr2sfm cameras.npz output.json -i /path/to/images
    pyav-idr2sfm cameras.npz output.json --sensor-width 23.5
"""

import argparse
import hashlib
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..image import get_image_dimensions
from ..sfmdata import SfMDataWrapper, WORLD_CORRECTION


def _fmt(v: float) -> str:
    """Format float with full precision for JSON serialization."""
    return f"{v:.17g}"


def generate_id(name: str) -> str:
    """Generate deterministic ID from name."""
    return str(
        int(hashlib.md5(name.encode(), usedforsecurity=False).hexdigest()[:8], 16)
        % 100000000
    )


def load_K_Rt_from_P(P: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Decompose projection matrix P into K (3x3) and Rt (4x4) in w2c form.

    Uses cv2.decomposeProjectionMatrix which returns K (with fx != fy from
    RQ decomposition), R (orthogonal), and camera center C. We then build
    the w2c Rt from R and C, and return K as-is.

    Note: cv2's RQ decomposition may produce fx != fy even for pinhole cameras.
    The pixelRatio (fy/fx) is stored in SfMData intrinsics to preserve this.

    Args:
        P: Projection matrix (4x4 or 3x4).

    Returns:
        Tuple of:
            - K: 3x3 intrinsics matrix (normalized so K[2,2] = 1)
            - Rt: 4x4 world-to-camera [R|t] matrix with orthogonal R
    """
    import cv2

    P_3x4 = P[:3, :4] if P.shape[0] == 4 else P

    out = cv2.decomposeProjectionMatrix(P_3x4)
    K_3x3 = out[0]
    R = out[1]
    t = out[2]

    # Normalize K so K[2,2] = 1
    if abs(K_3x3[2, 2]) < 1e-12:
        raise ValueError("Degenerate projection matrix: K[2,2] is zero")
    K_3x3 = K_3x3 / K_3x3[2, 2]

    # Ensure fx > 0 (cv2 RQ decomposition may produce negative diagonal)
    if K_3x3[0, 0] < 0:
        K_3x3[:, 0] *= -1
        R[0, :] *= -1
    if K_3x3[1, 1] < 0:
        K_3x3[:, 1] *= -1
        R[1, :] *= -1
    # Ensure det(R) = +1 (proper rotation, not reflection)
    if np.linalg.det(R) < 0:
        R *= -1

    # Camera center from homogeneous coordinates
    if abs(t[3]) < 1e-12:
        raise ValueError("Degenerate projection matrix: camera at infinity (t[3] ~ 0)")
    C = (t[:3] / t[3]).flatten()

    # R from cv2 is world-to-camera rotation (orthogonal)
    # Build Rt (4x4) in w2c form: Rt[:3,:3] = R, Rt[:3,3] = -R @ C
    Rt = np.eye(4, dtype=np.float64)
    Rt[:3, :3] = R
    Rt[:3, 3] = -R @ C

    return K_3x3.astype(np.float64), Rt


def _parse_view_indices(data) -> List[int]:
    """Parse actual view indices from npz keys (handles non-contiguous indices)."""
    indices = []
    for k in data.keys():
        m = re.match(r"^world_mat_(\d+)$", k)
        if m:
            indices.append(int(m.group(1)))
    return sorted(indices)


def load_idr_cameras(npz_path: Path) -> Dict:
    """Load cameras from IDR cameras.npz file.

    Args:
        npz_path: Path to cameras.npz

    Returns:
        Dict with keys:
            - K: list of 3x3 intrinsic matrices (per-view)
            - Rt: list of 4x4 [R|t] matrices in world-to-camera form
            - scale_mat: list of 4x4 scale matrices
            - indices: list of original view indices
            - n_views: number of views
    """
    data = np.load(str(npz_path))

    # Parse actual indices from keys (handles non-contiguous)
    indices = _parse_view_indices(data)
    if len(indices) == 0:
        raise ValueError(f"No world_mat_* keys found in {npz_path}")

    K_list = []
    Rt_list = []
    scale_mat_list = []

    for i in indices:
        P = data[f"world_mat_{i}"].astype(np.float64)

        # Load scale matrix (identity fallback)
        if f"scale_mat_{i}" in data:
            S = data[f"scale_mat_{i}"].astype(np.float64)
        else:
            S = np.eye(4, dtype=np.float64)

        # Check per-view if inverse matrices are available
        if f"camera_mat_inv_{i}" in data and f"camera_mat_{i}" in data:
            K_4x4 = data[f"camera_mat_{i}"].astype(np.float64)
            K_inv = data[f"camera_mat_inv_{i}"].astype(np.float64)
            if not np.allclose(K_inv[3, :], [0, 0, 0, 1]):
                raise ValueError(
                    f"camera_mat_inv_{i} bottom row must be [0,0,0,1], "
                    f"got {K_inv[3, :]}"
                )
            Rt = K_inv @ P  # 4x4 in w2c form
            K = K_4x4[:3, :3]  # Extract 3x3
        else:
            # Fallback: cv2.decomposeProjectionMatrix
            K, Rt = load_K_Rt_from_P(P)

        K_list.append(K)
        Rt_list.append(Rt)
        scale_mat_list.append(S)

    return {
        "K": K_list,
        "Rt": Rt_list,
        "scale_mat": scale_mat_list,
        "indices": indices,
        "n_views": len(indices),
    }


def idr_rotation_to_av(Rt: np.ndarray) -> List[str]:
    """Convert IDR [R|t] w2c to AliceVision cam2world rotation (9 floats, row-major).

    Steps:
        R_w2c = Rt[:3,:3]
        R_cam2world = R_w2c.T
        R_av = WORLD_CORRECTION @ R_cam2world
    """
    R_w2c = Rt[:3, :3]
    R_cam2world = R_w2c.T
    R_av = WORLD_CORRECTION @ R_cam2world
    return [_fmt(v) for v in R_av.flatten()]


def idr_center_to_av(Rt: np.ndarray) -> List[str]:
    """Convert IDR [R|t] w2c to AliceVision camera center (3 floats).

    Steps:
        center = -R_w2c.T @ t
        center_av = WORLD_CORRECTION @ center
    """
    R_w2c = Rt[:3, :3]
    t = Rt[:3, 3]
    center = -R_w2c.T @ t
    center_av = WORLD_CORRECTION @ center
    return [_fmt(v) for v in center_av]


def idr_intrinsics_to_av(
    K: np.ndarray, width: int, height: int, sensor_width: float
) -> Dict:
    """Convert IDR K matrix to AliceVision intrinsics dict.

    Args:
        K: 3x3 intrinsic matrix (may have fx != fy from RQ decomposition)
        width: Image width in pixels
        height: Image height in pixels
        sensor_width: Sensor width in mm

    Returns:
        Dict with focalLength, sensorWidth, pixelRatio, principalPoint, width, height
    """
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    if abs(fx) < 1e-12:
        raise ValueError("Degenerate intrinsics: fx is zero")

    focal_mm = fx * sensor_width / width
    pixel_ratio = fy / fx
    pp_offset_x = cx - width / 2.0
    pp_offset_y = cy - height / 2.0

    return {
        "focalLength": _fmt(focal_mm),
        "sensorWidth": _fmt(sensor_width),
        "pixelRatio": _fmt(pixel_ratio),
        "principalPoint": [_fmt(pp_offset_x), _fmt(pp_offset_y)],
        "width": str(width),
        "height": str(height),
    }


def find_images(images_folder: Path, n_views: int) -> List[Path]:
    """Find images in folder matching IDR view count.

    Strategy:
        1. Try IDR naming: {i:03d}.{ext} for each i in range(n_views)
        2. If that fails: list all image files sorted alphabetically

    Args:
        images_folder: Path to images folder
        n_views: Expected number of views

    Returns:
        List of n_views image Paths in order

    Raises:
        ValueError: If image count doesn't match n_views
    """
    extensions = [
        ".png", ".PNG", ".jpg", ".JPG", ".jpeg", ".JPEG", ".tif", ".TIF", ".exr", ".EXR"
    ]

    # Try IDR naming convention: 000.png, 001.png, ...
    idr_images = []
    for i in range(n_views):
        found = False
        for ext in extensions:
            candidate = images_folder / f"{i:03d}{ext}"
            if candidate.exists():
                idr_images.append(candidate)
                found = True
                break
        if not found:
            break

    if len(idr_images) == n_views:
        return idr_images

    # Fallback: list all image files sorted alphabetically
    all_images = []
    for ext in extensions:
        all_images.extend(images_folder.glob(f"*{ext}"))
    # Deduplicate (case variants may match same file on case-insensitive FS)
    all_images = sorted(set(all_images))

    if len(all_images) != n_views:
        raise ValueError(
            f"Found {len(all_images)} images but cameras.npz has {n_views} views"
        )

    return all_images


def convert_idr_to_sfmdata(
    npz_path: str,
    output_path: str,
    images_folder: Optional[str] = None,
    sensor_width: float = 36.0,
    save_scale_mats: bool = True,
) -> Dict:
    """Convert IDR cameras.npz to AliceVision SfMData JSON.

    Args:
        npz_path: Path to cameras.npz
        output_path: Output SfMData JSON path
        images_folder: Optional folder with source images
        sensor_width: Sensor width in mm (default: 36.0 full-frame)
        save_scale_mats: Save scale_mats.npz sidecar for round-trip

    Returns:
        SfMData dict
    """
    npz_path = Path(npz_path)
    output_path = Path(output_path)

    # Load IDR cameras
    idr = load_idr_cameras(npz_path)
    n_views = idr["n_views"]
    print(f"Loaded {n_views} views from {npz_path}")

    # Determine image dimensions
    image_paths = None
    if images_folder:
        images_folder = Path(images_folder)
        image_paths = find_images(images_folder, n_views)
        width, height = get_image_dimensions(str(image_paths[0]))
    else:
        # Infer from K: cx ~ width/2, cy ~ height/2
        # Note: approximate if principal point is offset from center
        K0 = idr["K"][0]
        width = int(round(2 * K0[0, 2]))
        height = int(round(2 * K0[1, 2]))
        if width <= 0 or height <= 0:
            raise ValueError(
                f"Cannot infer valid image dimensions from K "
                f"(got {width}x{height}). Provide --images-folder."
            )
        print(f"Inferred image dimensions from K: {width}x{height}")

    # Per-view intrinsics (each view may have slightly different K from decomposition)
    intrinsic_entries = []
    intrinsic_ids = []
    for i in range(n_views):
        intr_id = generate_id(f"intrinsic_{i}")
        intrinsic_ids.append(intr_id)
        intr_data = idr_intrinsics_to_av(idr["K"][i], width, height, sensor_width)

        intrinsic_entries.append({
            "intrinsicId": intr_id,
            "width": str(width),
            "height": str(height),
            "sensorWidth": _fmt(sensor_width),
            "sensorHeight": _fmt(sensor_width * height / width),
            "type": "pinhole",
            "initializationMode": "calibrated",
            "focalLength": intr_data["focalLength"],
            "pixelRatio": intr_data["pixelRatio"],
            "principalPoint": intr_data["principalPoint"],
            "distortionType": "none",
            "distortionParams": [],
            "locked": "false",
        })

    views = []
    poses = []

    for i in range(n_views):
        view_id = generate_id(f"view_{i}")
        pose_id = view_id

        # Image path
        if image_paths:
            img_path = str(image_paths[i].absolute())
        else:
            img_path = ""

        views.append({
            "viewId": view_id,
            "poseId": pose_id,
            "frameId": str(i),
            "intrinsicId": intrinsic_ids[i],
            "path": img_path,
            "width": str(width),
            "height": str(height),
        })

        poses.append({
            "poseId": pose_id,
            "pose": {
                "transform": {
                    "rotation": idr_rotation_to_av(idr["Rt"][i]),
                    "center": idr_center_to_av(idr["Rt"][i]),
                },
                "locked": "0",
            },
        })

    sfmdata = {
        "version": ["1", "2", "13"],
        "featuresFolders": [],
        "matchesFolders": [],
        "views": views,
        "intrinsics": intrinsic_entries,
        "poses": poses,
    }

    # Save via SfMDataWrapper
    output_path.parent.mkdir(parents=True, exist_ok=True)
    wrapper = SfMDataWrapper.from_dict(sfmdata)
    wrapper.save(str(output_path))
    print(f"Wrote {output_path}: {n_views} views, {n_views} poses")

    # Save scale_mats sidecar for round-trip
    if save_scale_mats:
        scale_mats_path = output_path.parent / "scale_mats.npz"
        scale_data = {
            f"scale_mat_{i}": idr["scale_mat"][i] for i in range(n_views)
        }
        np.savez(str(scale_mats_path), **scale_data)
        print(f"Wrote scale_mats.npz: {n_views} scale matrices")

    return sfmdata


def main():
    parser = argparse.ArgumentParser(
        description="Convert IDR cameras.npz to AliceVision SfMData",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  pyav-idr2sfm cameras.npz output.json
  pyav-idr2sfm cameras.npz output.json -i /path/to/images
  pyav-idr2sfm cameras.npz output.json --sensor-width 23.5
        """,
    )
    parser.add_argument("npz_path", help="Path to cameras.npz")
    parser.add_argument("output", help="Output SfMData JSON path")
    parser.add_argument(
        "--images-folder", "-i", help="Folder with source images"
    )
    parser.add_argument(
        "--sensor-width", type=float, default=36.0, help="Sensor width in mm (default: 36.0)"
    )
    parser.add_argument(
        "--no-save-scale-mats",
        action="store_true",
        help="Disable saving scale_mats.npz sidecar",
    )

    args = parser.parse_args()

    convert_idr_to_sfmdata(
        npz_path=args.npz_path,
        output_path=args.output,
        images_folder=args.images_folder,
        sensor_width=args.sensor_width,
        save_scale_mats=not args.no_save_scale_mats,
    )


if __name__ == "__main__":
    main()
