#!/usr/bin/env python3
"""
Convert COLMAP sparse reconstruction to AliceVision SfMData.

COLMAP text format:
- cameras.txt: CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]
- images.txt: IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME (2 lines per image)
- Quaternion: world-to-camera, Hamilton scalar-first
- Translation: world-to-camera: t = -R @ center
- Principal point: absolute pixel coordinates

AliceVision SfMData format:
- Rotation: cam2world (row-major, 9 values)
- Center: camera position in world
- focalLength: in millimeters
- principalPoint: offset from image center in pixels
- distortionType: "radialk3" with [k1, k2, 0] (ALWAYS — see F3 decision)

Usage:
    pyav-colmap2sfm <colmap_dir> <output.json> --sensor-width 36.0 --images-dir <dir>
"""

import argparse
import hashlib
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ..sfmdata import SfMDataWrapper, load_sfmdata, load_sfmdata_json

logger = logging.getLogger(__name__)


def generate_id(name: str) -> str:
    """Generate deterministic ID from name."""
    return str(int(hashlib.md5(name.encode()).hexdigest()[:8], 16) % 100000000)


def quat_to_matrix(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    """Convert quaternion to 3x3 rotation matrix.

    Hamilton scalar-first convention (qw, qx, qy, qz).
    Returns rotation matrix corresponding to the quaternion.
    """
    return np.array([
        [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)]
    ])


def _fmt(value: float) -> str:
    """Format float to string with full double precision."""
    return f"{value:.17g}"


def parse_cameras(cameras_path: Path) -> Dict[int, Dict]:
    """Parse COLMAP cameras.txt.

    Supported models: OPENCV (8 params), FULL_OPENCV (12 params), PINHOLE (4 params).

    Returns:
        Dict mapping camera_id to camera info dict.
    """
    cameras = {}
    with open(cameras_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            cam_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = [float(p) for p in parts[4:]]

            if model == 'OPENCV':
                if len(params) != 8:
                    raise ValueError(
                        f"Camera {cam_id}: OPENCV model expects 8 params, got {len(params)}"
                    )
                fx, fy, cx, cy, k1, k2, p1, p2 = params
                cameras[cam_id] = {
                    'model': model, 'width': width, 'height': height,
                    'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy,
                    'k1': k1, 'k2': k2, 'p1': p1, 'p2': p2, 'k3': 0.0,
                }
            elif model == 'FULL_OPENCV':
                if len(params) != 12:
                    raise ValueError(
                        f"Camera {cam_id}: FULL_OPENCV model expects 12 params, got {len(params)}"
                    )
                fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6 = params
                cameras[cam_id] = {
                    'model': model, 'width': width, 'height': height,
                    'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy,
                    'k1': k1, 'k2': k2, 'p1': p1, 'p2': p2, 'k3': k3,
                }
            elif model == 'PINHOLE':
                if len(params) != 4:
                    raise ValueError(
                        f"Camera {cam_id}: PINHOLE model expects 4 params, got {len(params)}"
                    )
                fx, fy, cx, cy = params
                cameras[cam_id] = {
                    'model': model, 'width': width, 'height': height,
                    'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy,
                    'k1': 0.0, 'k2': 0.0, 'p1': 0.0, 'p2': 0.0, 'k3': 0.0,
                }
            else:
                raise ValueError(
                    f"Camera {cam_id}: unsupported model '{model}'. "
                    "Supported: OPENCV, FULL_OPENCV, PINHOLE"
                )

    if not cameras:
        logger.warning("No cameras parsed from %s", cameras_path)
    return cameras


def parse_images(images_path: Path, cameras: Dict[int, Dict]) -> List[Dict]:
    """Parse COLMAP images.txt.

    Two lines per image:
        Line 1: IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
        Line 2: point2D list (skipped)

    Returns:
        List of image dicts with parsed extrinsics.
    """
    images = []
    with open(images_path, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line or line.startswith('#'):
            continue
        parts = line.split()
        if len(parts) < 10:
            continue

        image_id = int(parts[0])
        qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
        camera_id = int(parts[8])
        name = ' '.join(parts[9:])

        if camera_id not in cameras:
            raise ValueError(
                f"Image {image_id} ({name}): references camera_id {camera_id} "
                f"which is not in cameras.txt"
            )

        images.append({
            'image_id': image_id,
            'qw': qw, 'qx': qx, 'qy': qy, 'qz': qz,
            'tx': tx, 'ty': ty, 'tz': tz,
            'camera_id': camera_id,
            'name': name,
        })

        # Skip second line (2D points)
        if i < len(lines):
            i += 1

    if not images:
        logger.warning("No images parsed from %s", images_path)
    return images


def convert_colmap_to_sfmdata(
    colmap_dir: str,
    output_path: str,
    sensor_width: float = 36.0,
    images_dir: Optional[str] = None,
    reference_sfmdata: Optional[str] = None,
) -> Dict:
    """Convert COLMAP sparse reconstruction to AliceVision SfMData.

    Args:
        colmap_dir: Directory with cameras.txt, images.txt, points3D.txt
        output_path: Output SfMData JSON path
        sensor_width: Sensor width in mm
        images_dir: Directory containing actual images (to resolve paths)
        reference_sfmdata: Optional reference SfMData to match viewIds and extract metadata
    """
    colmap_dir = Path(colmap_dir)
    cameras_path = colmap_dir / 'cameras.txt'
    images_path = colmap_dir / 'images.txt'

    if not cameras_path.exists():
        raise FileNotFoundError(f"cameras.txt not found in {colmap_dir}")
    if not images_path.exists():
        raise FileNotFoundError(f"images.txt not found in {colmap_dir}")

    # Parse COLMAP files
    cameras = parse_cameras(cameras_path)
    images = parse_images(images_path, cameras)

    logger.info("Parsed %d cameras, %d images from COLMAP", len(cameras), len(images))

    # Load reference SfMData if provided
    ref_viewid_map = {}
    ref_sensor_width = None
    ref_serial_number = "0"
    ref_camera_make = "Unknown"
    ref_camera_model = "Unknown"

    if reference_sfmdata:
        try:
            ref_sfm = load_sfmdata(reference_sfmdata)
        except ImportError:
            # .sfm file that is actually JSON (Meshroom cameraInit.sfm)
            ref_sfm = SfMDataWrapper.from_dict(load_sfmdata_json(reference_sfmdata))
        ref_viewid_map = ref_sfm.get_viewid_by_image_name()
        logger.info("Loaded %d viewIds from reference SfMData", len(ref_viewid_map))

        # Extract metadata from reference
        ref_dict = ref_sfm.as_dict()
        ref_intrinsics = ref_dict.get('intrinsics', [])
        if ref_intrinsics:
            ref_intr = ref_intrinsics[0]
            ref_sensor_width = float(ref_intr.get('sensorWidth', sensor_width))
            ref_serial_number = ref_intr.get('serialNumber', '0')

        # Extract camera make/model from reference views metadata
        ref_views = ref_dict.get('views', [])
        if ref_views:
            meta = ref_views[0].get('metadata', {})
            ref_camera_make = meta.get('Make', meta.get('make', 'Unknown'))
            ref_camera_model = meta.get('Model', meta.get('model', 'Unknown'))

        # Cross-validate sensor_width (F9 fix)
        if ref_sensor_width is not None and abs(sensor_width - ref_sensor_width) > 0.1:
            logger.warning(
                "Sensor width mismatch: CLI=%.6f mm, reference=%.6f mm. "
                "Using reference value.",
                sensor_width, ref_sensor_width
            )
            sensor_width = ref_sensor_width

    serial_number = ref_serial_number if reference_sfmdata else "0"
    camera_make = ref_camera_make if reference_sfmdata else "Unknown"
    camera_model_name = ref_camera_model if reference_sfmdata else "Unknown"

    # Build SfMData structures
    views = []
    poses = []
    intrinsics_dict = {}

    # Resolve images directory
    if images_dir:
        images_dir_path = Path(images_dir)
    else:
        images_dir_path = None

    for img in images:
        cam = cameras[img['camera_id']]
        width = cam['width']
        height = cam['height']

        # --- Intrinsics conversion ---
        fx = cam['fx']
        fy = cam['fy']
        focal_mm = fx * sensor_width / width
        pixel_ratio = fy / fx if abs(fx) > 1e-15 else 1.0
        pp_offset_x = cam['cx'] - width / 2.0
        pp_offset_y = cam['cy'] - height / 2.0
        sensor_height = sensor_width * height / width

        # F3 fix: ALWAYS output radialk3 (eval_checkerboard.py handles it correctly
        # but has a buggy brown fallback where params[:5] puts k3 in p1 position)
        if cam['model'] == 'PINHOLE':
            dist_type = "none"
            dist_params = []
        else:
            dist_type = "radialk3"
            dist_params = [_fmt(cam['k1']), _fmt(cam['k2']), _fmt(cam['k3'])]
            if abs(cam['p1']) > 1e-9 or abs(cam['p2']) > 1e-9:
                logger.warning(
                    "Camera %d: dropping non-zero tangential distortion "
                    "p1=%s, p2=%s (OPENCV->radialk3)",
                    img['camera_id'], cam['p1'], cam['p2']
                )

        # Build intrinsic (one per COLMAP camera)
        intrinsic_key = str(img['camera_id'])
        if intrinsic_key not in intrinsics_dict:
            intrinsic_id = generate_id(f"colmap_cam_{img['camera_id']}")
            intrinsics_dict[intrinsic_key] = {
                'intrinsicId': intrinsic_id,
                'width': str(width),
                'height': str(height),
                'sensorWidth': _fmt(sensor_width),
                'sensorHeight': _fmt(sensor_height),
                'serialNumber': serial_number,
                'type': 'pinhole',
                'initializationMode': 'calibrated',
                'initialFocalLength': _fmt(focal_mm),
                'focalLength': _fmt(focal_mm),
                'pixelRatio': _fmt(pixel_ratio),
                'pixelRatioLocked': 'true',
                'principalPoint': [_fmt(pp_offset_x), _fmt(pp_offset_y)],
                'distortionType': dist_type,
                'distortionParams': dist_params if dist_params else [],
                'undistortionType': 'none',
                'undistortionOffset': ['0', '0'],
                'undistortionParams': [],
                'locked': 'false',
            }

        intrinsic_id = intrinsics_dict[intrinsic_key]['intrinsicId']

        # --- Extrinsics conversion ---
        R_w2c = quat_to_matrix(img['qw'], img['qx'], img['qy'], img['qz'])
        R_c2w = R_w2c.T
        center = -R_c2w @ np.array([img['tx'], img['ty'], img['tz']])
        rotation_flat = R_c2w.flatten()

        # --- View ID ---
        image_name = img['name']
        image_stem = Path(image_name).stem

        if image_stem in ref_viewid_map:
            view_id = ref_viewid_map[image_stem]
        else:
            view_id = generate_id(image_stem)

        pose_id = view_id

        # --- Image path ---
        if images_dir_path:
            image_path = str((images_dir_path / image_name).resolve())
        else:
            image_path = image_name

        # --- Frame ID ---
        frame_match = re.search(r'\d+', image_stem)
        frame_id = str(int(frame_match.group())) if frame_match else "0"

        views.append({
            'viewId': view_id,
            'poseId': pose_id,
            'frameId': frame_id,
            'intrinsicId': intrinsic_id,
            'path': image_path,
            'width': str(width),
            'height': str(height),
            'metadata': {'Make': camera_make, 'Model': camera_model_name},
        })

        poses.append({
            'poseId': pose_id,
            'pose': {
                'transform': {
                    'rotation': [_fmt(v) for v in rotation_flat],
                    'center': [_fmt(v) for v in center],
                },
                'locked': '0',
            },
        })

    sfmdata = {
        'version': ['1', '2', '13'],
        'featuresFolders': [],
        'matchesFolders': [],
        'views': views,
        'intrinsics': list(intrinsics_dict.values()),
        'poses': poses,
    }

    # Save using SfMDataWrapper
    wrapper = SfMDataWrapper.from_dict(sfmdata)
    wrapper.save(output_path)

    logger.info("Wrote %s: %d views, %d poses", output_path, len(views), len(poses))
    print(f"Wrote {output_path}: {len(views)} views, {len(poses)} poses")
    return sfmdata


def main():
    parser = argparse.ArgumentParser(
        description='Convert COLMAP sparse reconstruction to AliceVision SfMData',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  pyav-colmap2sfm ./sparse_txt/0 output.json --sensor-width 36.0 --images-dir ./images
  pyav-colmap2sfm ./sparse_txt/0 output.json --sensor-width 36.0 --images-dir ./images --reference ref.sfm
        """
    )
    parser.add_argument('colmap_dir', help='COLMAP text model directory (cameras.txt, images.txt)')
    parser.add_argument('output', help='Output SfMData JSON path')
    parser.add_argument('--sensor-width', type=float, default=36.0,
                        help='Sensor width in mm (default: 36.0)')
    parser.add_argument('--images-dir', help='Directory containing actual image files')
    parser.add_argument('--reference', '-r',
                        help='Reference SfMData to match viewIds and extract metadata')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    convert_colmap_to_sfmdata(
        args.colmap_dir,
        args.output,
        args.sensor_width,
        args.images_dir,
        args.reference,
    )


if __name__ == '__main__':
    main()
