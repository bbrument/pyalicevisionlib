#!/usr/bin/env python3
"""
Convert AliceVision SfMData to COLMAP sparse reconstruction (text format).

AliceVision SfMData format:
- Rotation: cam2world (row-major, 9 values)
- Center: camera position in world
- focalLength: in millimeters
- principalPoint: offset from image center in pixels

COLMAP text format:
- cameras.txt: CAMERA_ID FULL_OPENCV WIDTH HEIGHT fx fy cx cy k1 k2 p1 p2 k3 k4 k5 k6
- images.txt: IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
- Quaternion: world-to-camera, Hamilton scalar-first
- Translation: world-to-camera: t = -R @ center

Usage:
    pyav-sfm2colmap <sfmdata.json> <output_dir> [--sensor-width 36.0]
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ..sfmdata import load_sfmdata

logger = logging.getLogger(__name__)


def _fmt(value: float) -> str:
    """Format float to string with full double precision."""
    return f"{value:.17g}"


def matrix_to_quat(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to quaternion [qw, qx, qy, qz].

    Uses Shepperd's method. Enforces qw >= 0 (COLMAP convention).
    WARNING: Do NOT use scipy Rotation.as_quat() -- it returns scalar-last.
    """
    tr = np.trace(R)
    if tr > 0:
        s = 0.5 / np.sqrt(tr + 1.0)
        qw = 0.25 / s
        qx = (R[2, 1] - R[1, 2]) * s
        qy = (R[0, 2] - R[2, 0]) * s
        qz = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s

    q = np.array([qw, qx, qy, qz])
    if qw < 0:
        q = -q  # enforce qw >= 0
    return q / np.linalg.norm(q)  # normalize


def convert_sfmdata_to_colmap(
    sfmdata_path: str,
    output_dir: str,
    sensor_width_override: Optional[float] = None,
) -> None:
    """Convert AliceVision SfMData to COLMAP text format.

    Args:
        sfmdata_path: Path to SfMData JSON file
        output_dir: Output directory for cameras.txt, images.txt, points3D.txt
        sensor_width_override: Override sensor width in mm (if not in SfMData)
    """
    sfm = load_sfmdata(sfmdata_path)
    data = sfm.as_dict()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build intrinsics lookup
    intrinsics_dict = {}
    for intr in data.get('intrinsics', []):
        intrinsics_dict[intr['intrinsicId']] = intr

    # Build poses lookup
    poses_dict = {}
    for pose in data.get('poses', []):
        poses_dict[pose['poseId']] = pose['pose']['transform']

    # --- Write cameras.txt ---
    # Map intrinsicId to COLMAP camera_id (sequential)
    intrinsic_to_colmap_id = {}
    camera_lines = []

    for idx, (intr_id, intr) in enumerate(intrinsics_dict.items(), start=1):
        intrinsic_to_colmap_id[intr_id] = idx

        width = int(intr['width'])
        height = int(intr['height'])
        focal_mm = float(intr['focalLength'])
        sensor_w = float(intr.get('sensorWidth', sensor_width_override if sensor_width_override is not None else 36.0))
        pixel_ratio = float(intr.get('pixelRatio', '1'))

        fx = focal_mm * width / sensor_w
        fy = fx * pixel_ratio

        pp = intr.get('principalPoint', ['0', '0'])
        pp_offset_x = float(pp[0])
        pp_offset_y = float(pp[1])
        cx = width / 2.0 + pp_offset_x
        cy = height / 2.0 + pp_offset_y

        # Distortion: AV -> COLMAP FULL_OPENCV [k1,k2,p1,p2,k3,k4,k5,k6]
        dist_type = intr.get('distortionType', 'none')
        dist_params_raw = [float(p) for p in intr.get('distortionParams', [])]

        if dist_type == 'radialk3' and len(dist_params_raw) >= 3:
            k1, k2, k3 = dist_params_raw[0], dist_params_raw[1], dist_params_raw[2]
            colmap_dist = [k1, k2, 0.0, 0.0, k3, 0.0, 0.0, 0.0]
        elif dist_type == 'brown' and len(dist_params_raw) >= 5:
            k1, k2, k3, t1, t2 = dist_params_raw[:5]
            colmap_dist = [k1, k2, t1, t2, k3, 0.0, 0.0, 0.0]
        else:
            colmap_dist = [0.0] * 8

        params_str = ' '.join(_fmt(v) for v in [fx, fy, cx, cy] + colmap_dist)
        camera_lines.append(f"{idx} FULL_OPENCV {width} {height} {params_str}")

    cameras_path = output_dir / 'cameras.txt'
    with open(cameras_path, 'w') as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"# Number of cameras: {len(camera_lines)}\n")
        for line in camera_lines:
            f.write(line + '\n')

    # --- Write images.txt ---
    image_lines = []
    image_id = 1

    for view in data.get('views', []):
        pose_id = view.get('poseId')
        intr_id = view.get('intrinsicId')

        if pose_id not in poses_dict or intr_id not in intrinsic_to_colmap_id:
            continue

        transform = poses_dict[pose_id]
        colmap_cam_id = intrinsic_to_colmap_id[intr_id]

        # Parse rotation (cam2world, row-major)
        rot_flat = [float(r) for r in transform['rotation']]
        R_c2w = np.array(rot_flat).reshape(3, 3)
        R_w2c = R_c2w.T

        # Parse center
        center = np.array([float(c) for c in transform['center']])

        # Convert to COLMAP convention
        t = -R_w2c @ center
        quat = matrix_to_quat(R_w2c)  # [qw, qx, qy, qz]

        # Image name (just filename, not full path)
        image_path = view.get('path', '')
        image_name = Path(image_path).name

        quat_str = ' '.join(_fmt(v) for v in quat)
        t_str = ' '.join(_fmt(v) for v in t)
        image_lines.append(f"{image_id} {quat_str} {t_str} {colmap_cam_id} {image_name}")
        image_id += 1

    images_path = output_dir / 'images.txt'
    with open(images_path, 'w') as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {len(image_lines)}\n")
        for line in image_lines:
            f.write(line + '\n')
            f.write('\n')  # empty second line (no 2D points)

    # --- Write points3D.txt (empty) ---
    points_path = output_dir / 'points3D.txt'
    with open(points_path, 'w') as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write("# Number of points: 0\n")

    logger.info(
        "Wrote COLMAP model to %s: %d cameras, %d images",
        output_dir, len(camera_lines), len(image_lines)
    )
    print(f"Wrote COLMAP model to {output_dir}: {len(camera_lines)} cameras, {len(image_lines)} images")


def main():
    parser = argparse.ArgumentParser(
        description='Convert AliceVision SfMData to COLMAP text format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  pyav-sfm2colmap sfmdata.json ./colmap_output
  pyav-sfm2colmap sfmdata.json ./colmap_output --sensor-width 36.0
        """
    )
    parser.add_argument('sfmdata', help='Input SfMData JSON path')
    parser.add_argument('output_dir', help='Output directory for COLMAP text files')
    parser.add_argument('--sensor-width', type=float, default=None,
                        help='Override sensor width in mm (uses SfMData value if not specified)')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    convert_sfmdata_to_colmap(
        args.sfmdata,
        args.output_dir,
        args.sensor_width,
    )


if __name__ == '__main__':
    main()
