#!/usr/bin/env python3
"""
Convert RealityCapture XMP to AliceVision SfMData.

RealityCapture XMP format:
- Rotation: world2cam (row-major)
- Position: camera center in world
- PrincipalPointU/V: normalized by max(width, height)
- DistortionCoeficients: [k1, k2, k3, k4, t1, t2]

AliceVision SfMData format:
- Rotation: cam2world (row-major)
- Center: camera center in world
- PrincipalPoint: offset from image center in pixels

Usage:
    pyav-rc2sfm <xmp_folder> <images_folder> <output.json>
"""

import argparse
import hashlib
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

from ..sfmdata import load_sfmdata, SfMDataWrapper, WORLD_CORRECTION
from ..image import get_image_dimensions


def parse_xmp_file(xmp_path: str) -> Dict:
    """Parse RealityCapture XMP file."""
    with open(xmp_path, 'r') as f:
        content = f.read()
    
    root = ET.fromstring(content)
    ns = {'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#'}
    xcr_ns = '{http://www.capturingreality.com/ns/xcr/1.1#}'
    
    desc = root.find('.//rdf:Description', ns)
    if desc is None:
        raise ValueError(f"No rdf:Description found in {xmp_path}")
    
    data = {
        'focal_length_35mm': float(desc.get(f'{xcr_ns}FocalLength35mm', 0)),
        'principal_point_u': float(desc.get(f'{xcr_ns}PrincipalPointU', 0)),
        'principal_point_v': float(desc.get(f'{xcr_ns}PrincipalPointV', 0)),
        'distortion_model': desc.get(f'{xcr_ns}DistortionModel', 'none'),
    }
    
    for child in desc:
        tag = child.tag.replace(xcr_ns, '')
        if tag == 'Rotation':
            data['rotation'] = [float(x) for x in child.text.strip().split()]
        elif tag == 'Position':
            data['position'] = [float(x) for x in child.text.strip().split()]
        elif tag == 'DistortionCoeficients':
            data['distortion_coefficients'] = [float(x) for x in child.text.strip().split()]
    return data


def find_image(images_folder: Path, base_name: str) -> Optional[Path]:
    """Find image matching base_name in folder."""
    image_base = base_name
    for suffix in ['_masked', '_undistorted', '_crop']:
        if image_base.endswith(suffix):
            image_base = image_base[:-len(suffix)]
    
    extensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.exr',
                  '.ARW', '.arw', '.CR2', '.cr2', '.NEF', '.nef', '.DNG', '.dng']
    
    for name in [base_name, image_base]:
        for ext in extensions:
            candidate = images_folder / f"{name}{ext}"
            if candidate.exists():
                return candidate
    
    for ext in extensions:
        for candidate in images_folder.glob(f"*{ext}"):
            if image_base in candidate.stem:
                return candidate
    
    return None


def generate_id(name: str) -> str:
    """Generate deterministic ID from name."""
    return str(int(hashlib.md5(name.encode()).hexdigest()[:8], 16) % 100000000)


def rc_rotation_to_av(rc_rotation: List[float]) -> List[str]:
    """
    Convert RC rotation (world2cam) to AV rotation (cam2world).
    
    Also applies world coordinate correction (flip Y and Z) to match
    AliceVision/Meshroom conventions.
    """
    R = np.array(rc_rotation).reshape(3, 3)
    # Transpose: world2cam -> cam2world
    R_cam2world = R.T
    # Apply world correction: R_corrected = WORLD_CORRECTION @ R_cam2world
    R_corrected = WORLD_CORRECTION @ R_cam2world
    return [str(v) for v in R_corrected.flatten()]


def rc_position_to_av(rc_position: List[float]) -> List[str]:
    """
    Convert RC position to AV center.
    
    Applies world coordinate correction (flip Y and Z) to match
    AliceVision/Meshroom conventions.
    """
    center = np.array(rc_position)
    # Apply world correction: center_corrected = WORLD_CORRECTION @ center
    center_corrected = WORLD_CORRECTION @ center
    return [str(v) for v in center_corrected]


def convert_principal_point(pp_u: float, pp_v: float, width: int, height: int) -> Tuple[float, float]:
    """Convert RC principal point (normalized) to AV (offset in pixels)."""
    max_dim = max(width, height)
    pp_x = pp_u * max_dim
    pp_y = pp_v * max_dim
    return pp_x, pp_y


def convert_distortion(coeffs: List[float]) -> Tuple[str, List[str]]:
    """Convert RC distortion to AV format."""
    if len(coeffs) < 3:
        return 'none', []
    
    k1, k2, k3 = coeffs[:3]
    t1 = coeffs[4] if len(coeffs) > 4 else 0
    t2 = coeffs[5] if len(coeffs) > 5 else 0
    
    if abs(t1) > 1e-10 or abs(t2) > 1e-10:
        return 'brown', [str(k1), str(k2), str(k3), str(t1), str(t2)]
    else:
        return 'radialk3', [str(k1), str(k2), str(k3)]


def convert_rc_to_sfmdata(
    xmp_folder: str,
    images_folder: str,
    output_path: str,
    sensor_width: float = 36.0,
    sensor_height: float = 24.0,
    camera_make: str = "Unknown",
    camera_model: str = "Unknown",
    serial_number: str = "0",
    reference_sfmdata: Optional[str] = None
) -> Dict:
    """Convert RealityCapture XMP files to AliceVision SfMData.
    
    Args:
        xmp_folder: Folder containing XMP files
        images_folder: Folder containing images
        output_path: Output SfMData JSON path
        sensor_width: Sensor width in mm
        sensor_height: Sensor height in mm
        camera_make: Camera make string
        camera_model: Camera model string
        serial_number: Camera serial number
        reference_sfmdata: Optional reference SfMData to match viewIds by image name
    """
    xmp_folder = Path(xmp_folder)
    images_folder = Path(images_folder)
    
    # Load reference viewId mapping if provided
    ref_viewid_map = {}
    if reference_sfmdata:
        ref_sfm = load_sfmdata(reference_sfmdata)
        ref_viewid_map = ref_sfm.get_viewid_by_image_name()
        print(f"Loaded {len(ref_viewid_map)} viewIds from reference SfMData")
    
    xmp_files = sorted(xmp_folder.glob('*.xmp'))
    if not xmp_files:
        raise ValueError(f"No XMP files found in {xmp_folder}")
    
    print(f"Found {len(xmp_files)} XMP files")
    
    views = []
    poses = []
    intrinsics_dict = {}
    
    for xmp_path in xmp_files:
        xmp_data = parse_xmp_file(str(xmp_path))
        base_name = xmp_path.stem
        
        image_path = find_image(images_folder, base_name)
        if image_path is None:
            print(f"Warning: No image found for {xmp_path.name}")
            continue
        
        width, height = get_image_dimensions(str(image_path))
        
        # Use reference viewId if available, otherwise generate one
        image_stem = image_path.stem
        if image_stem in ref_viewid_map:
            view_id = ref_viewid_map[image_stem]
        elif base_name in ref_viewid_map:
            view_id = ref_viewid_map[base_name]
        else:
            view_id = generate_id(base_name)
        pose_id = view_id
        
        focal_35mm = xmp_data['focal_length_35mm']
        focal_mm = focal_35mm * sensor_width / 36.0
        
        pp_x, pp_y = convert_principal_point(
            xmp_data['principal_point_u'],
            xmp_data['principal_point_v'],
            width, height
        )
        
        dist_coeffs = xmp_data.get('distortion_coefficients', [])
        dist_type, dist_params = convert_distortion(dist_coeffs)
        
        intrinsic_key = f"{focal_mm:.4f}_{serial_number}"
        if intrinsic_key not in intrinsics_dict:
            intrinsics_dict[intrinsic_key] = {
                'intrinsicId': generate_id(intrinsic_key),
                'width': str(width),
                'height': str(height),
                'sensorWidth': str(sensor_width),
                'sensorHeight': str(sensor_height),
                'serialNumber': serial_number,
                'type': 'pinhole',
                'initializationMode': 'calibrated',
                'initialFocalLength': str(focal_mm),
                'focalLength': str(focal_mm),
                'pixelRatio': '1',
                'pixelRatioLocked': 'true',
                'offsetLocked': 'false',
                'scaleLocked': 'false',
                'principalPoint': [str(pp_x), str(pp_y)],
                'distortionLocked': 'false',
                'distortionInitializationMode': 'none',
                'distortionType': dist_type,
                'distortionParams': dist_params if dist_params else ['0', '0', '0'],
                'undistortionOffset': ['0', '0'],
                'undistortionParams': [],
                'undistortionType': 'none',
                'locked': 'false'
            }
        
        intrinsic_id = intrinsics_dict[intrinsic_key]['intrinsicId']
        
        frame_match = re.search(r'\d+', base_name)
        views.append({
            'viewId': view_id,
            'poseId': pose_id,
            'frameId': str(int(frame_match.group()) if frame_match else 0),
            'intrinsicId': intrinsic_id,
            'path': str(image_path.absolute()),
            'width': str(width),
            'height': str(height),
            'metadata': {'Make': camera_make, 'Model': camera_model}
        })
        
        if 'rotation' in xmp_data and 'position' in xmp_data:
            poses.append({
                'poseId': pose_id,
                'pose': {
                    'transform': {
                        'rotation': rc_rotation_to_av(xmp_data['rotation']),
                        'center': rc_position_to_av(xmp_data['position'])
                    },
                    'locked': '0'
                }
            })
    
    sfmdata = {
        'version': ['1', '2', '13'],
        'featuresFolders': [],
        'matchesFolders': [],
        'views': views,
        'intrinsics': list(intrinsics_dict.values()),
        'poses': poses
    }
    
    # Use wrapper to save - this normalizes the structure via pyalicevision
    wrapper = SfMDataWrapper.from_dict(sfmdata)
    wrapper.save(output_path)
    
    print(f"Wrote {output_path}: {len(views)} views, {len(poses)} poses")
    return sfmdata


def main():
    parser = argparse.ArgumentParser(
        description='Convert RealityCapture XMP to AliceVision SfMData',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  pyav-rc2sfm ./xmp_files ./images output.json
  pyav-rc2sfm ./xmp_files ./images output.json --sensor-width 23.5
  pyav-rc2sfm ./xmp_files ./images output.json --reference /path/to/ref_sfmdata.json
        """
    )
    parser.add_argument('xmp_folder', help='Folder with XMP files')
    parser.add_argument('images_folder', help='Folder with images')
    parser.add_argument('output', help='Output SfMData JSON')
    parser.add_argument('--reference', '-r', help='Reference SfMData to match viewIds by image name')
    parser.add_argument('--sensor-width', type=float, default=36.0, help='Sensor width mm')
    parser.add_argument('--sensor-height', type=float, default=24.0, help='Sensor height mm')
    parser.add_argument('--camera-make', default='Unknown', help='Camera make')
    parser.add_argument('--camera-model', default='Unknown', help='Camera model')
    parser.add_argument('--serial-number', default='0', help='Serial number')
    
    args = parser.parse_args()
    
    convert_rc_to_sfmdata(
        args.xmp_folder,
        args.images_folder,
        args.output,
        args.sensor_width,
        args.sensor_height,
        args.camera_make,
        args.camera_model,
        args.serial_number,
        args.reference
    )


if __name__ == '__main__':
    main()
