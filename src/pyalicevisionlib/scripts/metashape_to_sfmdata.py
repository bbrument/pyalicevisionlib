#!/usr/bin/env python3
"""
Convert Agisoft Metashape XML to AliceVision SfMData.

Metashape XML format:
- Camera transforms: 4x4 cam2world matrices (row-major), local to chunk
- Component transform: rotation (3x3), translation (3), scale (1)
- Sensor calibration: f (pixels), cx/cy (pixel offset from center), k1-k3, p1-p2

AliceVision SfMData format:
- Rotation: cam2world (row-major)
- Center: camera center in world
- PrincipalPoint: offset from image center in pixels
- Distortion: Brown model [k1, k2, k3, p1, p2]

Usage:
    pyav-metashape2sfm <metashape.xml> <output.json> [--sensor-width 36.0]
"""

import argparse
import hashlib
import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.linalg import polar

from ..sfmdata import SfMDataWrapper, WORLD_CORRECTION_4x4, load_sfmdata

logger = logging.getLogger(__name__)


def generate_id(name: str) -> str:
    """Generate deterministic ID from name."""
    return str(int(hashlib.md5(name.encode()).hexdigest()[:8], 16) % 100000000)


def parse_metashape_xml(xml_path: str) -> ET.Element:
    """Parse Metashape XML file and return the chunk element.

    Validates the expected structure: <document> -> <chunk> with <sensors> and <cameras>.

    Raises:
        ValueError: If the XML structure is invalid or missing required elements.
    """
    try:
        # Note: ET.parse does not resolve external entities by default in Python 3.
        # For additional protection against XXE, install and use defusedxml.
        tree = ET.parse(xml_path)
    except ET.ParseError as e:
        raise ValueError(f"Failed to parse XML file {xml_path}: {e}")

    root = tree.getroot()
    if root.tag != 'document':
        raise ValueError(f"Expected root tag 'document', got '{root.tag}'")

    chunk = root.find('chunk')
    if chunk is None:
        raise ValueError("No <chunk> element found in Metashape XML")

    if chunk.find('.//sensors') is None:
        raise ValueError("No <sensors> element found in Metashape XML")

    if chunk.find('.//cameras') is None:
        raise ValueError("No <cameras> element found in Metashape XML")

    return chunk


def extract_component_transforms(chunk: ET.Element) -> Dict[str, Tuple[np.ndarray, float]]:
    """Extract component (chunk) transforms from XML.

    Returns:
        Dict keyed by component id string, each value is (transform_4x4, scale).
        Components without a <transform> element get identity transform with scale=1.0.
    """
    components = {}
    for comp in chunk.findall('.//component'):
        comp_id = comp.get('id', '0')
        transform_el = comp.find('transform')
        if transform_el is not None:
            rot_el = transform_el.find('rotation')
            trans_el = transform_el.find('translation')
            scale_el = transform_el.find('scale')

            if rot_el is not None and trans_el is not None and scale_el is not None:
                rot_text = rot_el.text if rot_el.text else rot_el.get('#text', '')
                trans_text = trans_el.text if trans_el.text else trans_el.get('#text', '')
                scale_text = scale_el.text if scale_el.text else scale_el.get('#text', '')

                rotation = np.array([float(x) for x in rot_text.split()]).reshape(3, 3)
                translation = np.array([float(x) for x in trans_text.split()])
                scale = float(scale_text)

                transform_4x4 = np.eye(4)
                transform_4x4[:3, :3] = scale * rotation
                transform_4x4[:3, 3] = translation
                components[comp_id] = (transform_4x4, scale)
            else:
                components[comp_id] = (np.eye(4), 1.0)
        else:
            components[comp_id] = (np.eye(4), 1.0)

    return components


def extract_sensors(chunk: ET.Element) -> Dict[str, Dict]:
    """Extract sensor calibration data from XML.

    Returns:
        Dict keyed by sensor id string with calibration parameters.
    """
    sensors = {}
    for sensor in chunk.findall('.//sensor'):
        sensor_id = sensor.get('id')
        res = sensor.find('resolution')
        if res is None:
            logger.warning("Skipping sensor %s: no <resolution> element", sensor_id)
            continue

        width = int(res.get('width'))
        height = int(res.get('height'))

        calib = sensor.find('calibration')
        if calib is None:
            logger.warning("Skipping sensor %s: no <calibration> element", sensor_id)
            continue

        f_el = calib.find('f')
        f = float(f_el.text) if f_el is not None else 0.0

        cx = float(calib.findtext('cx', '0'))
        cy = float(calib.findtext('cy', '0'))
        k1 = float(calib.findtext('k1', '0'))
        k2 = float(calib.findtext('k2', '0'))
        k3 = float(calib.findtext('k3', '0'))
        p1 = float(calib.findtext('p1', '0'))
        p2 = float(calib.findtext('p2', '0'))

        sensors[sensor_id] = {
            'width': width,
            'height': height,
            'f': f,
            'cx': cx,
            'cy': cy,
            'k1': k1,
            'k2': k2,
            'k3': k3,
            'p1': p1,
            'p2': p2,
            'label': sensor.get('label', f'Sensor {sensor_id}'),
        }

    return sensors


def convert_metashape_to_sfmdata(
    xml_path: str,
    output_path: str,
    sensor_width: float = 36.0,
    sensor_height: float = 24.0,
    images_folder: Optional[str] = None,
    reference_sfmdata: Optional[str] = None,
) -> Dict:
    """Convert Metashape XML to AliceVision SfMData.

    Args:
        xml_path: Path to Metashape XML file
        output_path: Output SfMData JSON path
        sensor_width: Sensor width in mm (default 36.0)
        sensor_height: Sensor height in mm (default 24.0)
        images_folder: Optional folder containing images
        reference_sfmdata: Optional reference SfMData to match viewIds by image name
    """
    chunk = parse_metashape_xml(xml_path)
    sensors = extract_sensors(chunk)
    components = extract_component_transforms(chunk)

    # Load reference viewId mapping if provided
    ref_viewid_map = {}
    if reference_sfmdata:
        ref_sfm = load_sfmdata(reference_sfmdata)
        ref_viewid_map = ref_sfm.get_viewid_by_image_name()
        logger.info("Loaded %d viewIds from reference SfMData", len(ref_viewid_map))

    # Detect image extension from images_folder if provided (sorted for determinism)
    image_ext = 'jpg'
    if images_folder:
        images_path = Path(images_folder)
        if images_path.exists():
            image_extensions = ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.exr', '.arw', '.cr2', '.nef', '.dng')
            for f in sorted(images_path.iterdir()):
                if f.is_file() and f.suffix.lower() in image_extensions:
                    image_ext = f.suffix.lstrip('.')
                    break

    # Build intrinsics from sensors
    intrinsics_dict = {}
    sensor_to_intrinsic_id = {}
    for sensor_id, sensor_data in sensors.items():
        w = sensor_data['width']
        h = sensor_data['height']
        f_px = sensor_data['f']
        focal_mm = f_px * sensor_width / w

        intrinsic_id = generate_id(f"sensor_{sensor_id}")
        sensor_to_intrinsic_id[sensor_id] = intrinsic_id

        intrinsics_dict[intrinsic_id] = {
            'intrinsicId': intrinsic_id,
            'width': str(w),
            'height': str(h),
            'sensorWidth': str(sensor_width),
            'sensorHeight': str(sensor_height),
            'serialNumber': '-1',
            'type': 'pinhole',
            'initializationMode': 'calibrated',
            'initialFocalLength': f'{focal_mm:.15g}',
            'focalLength': f'{focal_mm:.15g}',
            'pixelRatio': '1',
            'pixelRatioLocked': 'true',
            'offsetLocked': 'false',
            'scaleLocked': 'false',
            'principalPoint': [f'{sensor_data["cx"]:.15g}', f'{sensor_data["cy"]:.15g}'],
            'distortionLocked': 'false',
            'distortionInitializationMode': 'none',
            'distortionType': 'brown',
            'distortionParams': [
                f'{sensor_data["k1"]:.15g}',
                f'{sensor_data["k2"]:.15g}',
                f'{sensor_data["k3"]:.15g}',
                f'{sensor_data["p1"]:.15g}',
                f'{sensor_data["p2"]:.15g}',
            ],
            'undistortionOffset': ['0', '0'],
            'undistortionParams': [],
            'undistortionType': 'none',
            'locked': 'false',
        }

    # Process cameras
    views = []
    poses = []

    for camera in chunk.findall('.//camera'):
        transform_el = camera.find('transform')
        if transform_el is None or transform_el.text is None:
            continue  # Skip unaligned cameras

        label = camera.get('label', '')
        sensor_id = camera.get('sensor_id', '0')
        component_id = camera.get('component_id', '0')

        if sensor_id not in sensor_to_intrinsic_id:
            continue

        # Parse 4x4 camera transform (row-major)
        cam_local = np.array([float(x) for x in transform_el.text.split()]).reshape(4, 4)

        # Get component transform
        comp_transform, comp_scale = components.get(component_id, (np.eye(4), 1.0))

        # Apply component transform: world_pose = component_transform @ camera_local
        world_pose = comp_transform @ cam_local

        # Apply world correction for AliceVision convention
        av_pose = WORLD_CORRECTION_4x4 @ world_pose

        # Extract rotation and center
        R = av_pose[:3, :3]
        if comp_scale != 1.0:
            R = R / comp_scale
        # Orthogonalize to ensure valid rotation matrix (handles numerical errors)
        R, _ = polar(R)
        if np.linalg.det(R) < 0:
            R = -R
        center = av_pose[:3, 3]

        # Generate or match viewId
        if label in ref_viewid_map:
            view_id = ref_viewid_map[label]
        else:
            view_id = generate_id(label)
        pose_id = view_id

        intrinsic_id = sensor_to_intrinsic_id[sensor_id]

        # Build image path
        if images_folder:
            img_path = str(Path(images_folder) / f"{label}.{image_ext}")
        else:
            img_path = f"{label}.{image_ext}"

        sensor_data = sensors[sensor_id]
        views.append({
            'viewId': view_id,
            'poseId': pose_id,
            'frameId': str(len(views)),
            'intrinsicId': intrinsic_id,
            'path': img_path,
            'width': str(sensor_data['width']),
            'height': str(sensor_data['height']),
            'metadata': {},
        })

        poses.append({
            'poseId': pose_id,
            'pose': {
                'transform': {
                    'rotation': [f'{v:.15g}' for v in R.flatten()],
                    'center': [f'{v:.15g}' for v in center],
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

    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    wrapper = SfMDataWrapper.from_dict(sfmdata)
    wrapper.save(output_path)

    logger.info("Converted %d cameras from Metashape XML", len(views))
    logger.info("  Sensors: %d, Intrinsics: %d", len(sensors), len(intrinsics_dict))
    logger.info("  Output: %s", output_path)

    return sfmdata


def main():
    parser = argparse.ArgumentParser(
        description='Convert Agisoft Metashape XML to AliceVision SfMData',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  pyav-metashape2sfm metashape.xml output.json
  pyav-metashape2sfm metashape.xml output.json --sensor-width 23.5
  pyav-metashape2sfm metashape.xml output.json --images-folder ./images
  pyav-metashape2sfm metashape.xml output.json --reference ref_sfmdata.json
        """,
    )
    parser.add_argument('xml_path', help='Input Metashape XML file')
    parser.add_argument('output', help='Output SfMData JSON file')
    parser.add_argument('--images-folder', '-i', help='Folder containing images')
    parser.add_argument('--sensor-width', type=float, default=36.0, help='Sensor width in mm (default: 36.0)')
    parser.add_argument('--sensor-height', type=float, default=24.0, help='Sensor height in mm (default: 24.0)')
    parser.add_argument('--reference', '-r', help='Reference SfMData to match viewIds by image name')

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    convert_metashape_to_sfmdata(
        args.xml_path,
        args.output,
        args.sensor_width,
        args.sensor_height,
        args.images_folder,
        args.reference,
    )


if __name__ == '__main__':
    main()
