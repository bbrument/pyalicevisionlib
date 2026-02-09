#!/usr/bin/env python3
"""
Convert AliceVision SfMData to RealityCapture XMP.

AliceVision SfMData format:
- Rotation: cam2world (row-major)
- Center: camera center in world
- PrincipalPoint: offset from image center in pixels

RealityCapture XMP format:
- Rotation: world2cam (row-major)
- Position: camera center in world
- PrincipalPointU/V: normalized by max(width, height)
- DistortionCoeficients: [k1, k2, k3, k4, t1, t2]

Usage:
    pyav-sfm2rc <sfmdata.json> <output_folder> [--images-folder <name>]
"""

import argparse
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

from ..sfmdata import SfMDataWrapper, WORLD_CORRECTION


# =============================================================================
# XMP Template
# =============================================================================

XMP_TEMPLATE = '''<x:xmpmeta xmlns:x="adobe:ns:meta/">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about=""
    xmlns:xcr="http://www.capturingreality.com/ns/xcr/1.1#"
    xcr:Version="3"
    xcr:PosePrior="locked"
    xcr:Coordinates="absolute"
    xcr:DistortionModel="{distortion_model}"
    xcr:FocalLength35mm="{focal_35mm}"
    xcr:Skew="0"
    xcr:AspectRatio="1"
    xcr:PrincipalPointU="{pp_u}"
    xcr:PrincipalPointV="{pp_v}"
    xcr:CalibrationPrior="locked"
    xcr:CalibrationGroup="-1"
    xcr:DistortionGroup="-1"
    xcr:InTexturing="1"
    xcr:InMeshing="1">
   <xcr:Rotation>{rotation}</xcr:Rotation>
   <xcr:Position>{position}</xcr:Position>
   <xcr:DistortionCoeficients>{distortion}</xcr:DistortionCoeficients>
  </rdf:Description>
 </rdf:RDF>
</x:xmpmeta>
'''


# =============================================================================
# Conversion functions (AV -> RC) - Inverse of rc_to_sfmdata.py
# =============================================================================

def av_rotation_to_rc(av_rotation: np.ndarray) -> List[float]:
    """
    Convert AV rotation (cam2world) to RC rotation (world2cam).
    
    Inverse of rc_rotation_to_av:
    - In rc_to_sfmdata: R_corrected = WORLD_CORRECTION @ R.T
    - Inverse: R_original = (WORLD_CORRECTION @ R_corrected).T
    
    Since WORLD_CORRECTION is symmetric (diagonal with 1, -1, -1):
        WORLD_CORRECTION^-1 = WORLD_CORRECTION
    
    Args:
        av_rotation: 3x3 cam2world rotation matrix from AV (already corrected)
        
    Returns:
        List of 9 floats for RC world2cam rotation (row-major)
    """
    # Undo world correction: R_cam2world_original = WORLD_CORRECTION @ R_corrected
    R_cam2world_original = WORLD_CORRECTION @ av_rotation
    # Transpose to get world2cam
    R_world2cam = R_cam2world_original.T
    return R_world2cam.flatten().tolist()


def av_position_to_rc(av_center: np.ndarray) -> List[float]:
    """
    Convert AV center to RC position.
    
    Inverse of rc_position_to_av:
    - In rc_to_sfmdata: center_corrected = WORLD_CORRECTION @ center
    - Inverse: center_original = WORLD_CORRECTION @ center_corrected
    
    Args:
        av_center: Camera center from AV (already corrected)
        
    Returns:
        List of 3 floats for RC position
    """
    # WORLD_CORRECTION is its own inverse
    position = WORLD_CORRECTION @ av_center
    return position.tolist()


def convert_principal_point_to_rc(
    pp_x: float, 
    pp_y: float, 
    width: int, 
    height: int
) -> Tuple[float, float]:
    """
    Convert AV principal point (offset in pixels) to RC (normalized).
    
    Inverse of convert_principal_point in rc_to_sfmdata.py:
    - In rc_to_sfmdata: pp_x = pp_u * max_dim
    - Inverse: pp_u = pp_x / max_dim
    
    Args:
        pp_x: Principal point X offset from image center (pixels)
        pp_y: Principal point Y offset from image center (pixels)
        width: Image width
        height: Image height
        
    Returns:
        Tuple of (pp_u, pp_v) normalized values
    """
    max_dim = max(width, height)
    pp_u = pp_x / max_dim
    pp_v = pp_y / max_dim
    return pp_u, pp_v


def convert_distortion_to_rc(
    dist_type: str, 
    dist_params: List[float]
) -> Tuple[str, List[float]]:
    """
    Convert AV distortion to RC format.
    
    Inverse of convert_distortion in rc_to_sfmdata.py.
    
    Args:
        dist_type: AV distortion type ('none', 'radialk3', 'brown')
        dist_params: AV distortion parameters
        
    Returns:
        Tuple of (rc_distortion_model, [k1, k2, k3, k4, t1, t2])
    """
    if dist_type == 'none' or not dist_params:
        return 'none', [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    # Get k1, k2, k3 (first 3 params)
    k1 = dist_params[0] if len(dist_params) > 0 else 0.0
    k2 = dist_params[1] if len(dist_params) > 1 else 0.0
    k3 = dist_params[2] if len(dist_params) > 2 else 0.0
    k4 = 0.0  # RC has k4, AV typically doesn't use it
    
    # Get tangential params for brown model
    if dist_type == 'brown' and len(dist_params) >= 5:
        t1 = dist_params[3]
        t2 = dist_params[4]
        return 'brown3t2', [k1, k2, k3, k4, t1, t2]
    else:
        return 'brown3', [k1, k2, k3, k4, 0.0, 0.0]


def focal_mm_to_35mm(focal_mm: float, sensor_width: float) -> float:
    """
    Convert focal length in mm to 35mm equivalent.
    
    Inverse of the conversion in rc_to_sfmdata.py:
    - In rc_to_sfmdata: focal_mm = focal_35mm * sensor_width / 36.0
    - Inverse: focal_35mm = focal_mm * 36.0 / sensor_width
    
    Args:
        focal_mm: Focal length in mm
        sensor_width: Sensor width in mm
        
    Returns:
        Focal length in 35mm equivalent
    """
    return focal_mm * 36.0 / sensor_width


# =============================================================================
# XMP Generation
# =============================================================================

def generate_xmp_content(
    rotation: List[float],
    position: List[float],
    focal_35mm: float,
    pp_u: float,
    pp_v: float,
    distortion_model: str = "brown3",
    distortion: List[float] = None
) -> str:
    """
    Generate XMP file content for RealityCapture.
    
    Args:
        rotation: 9 floats for world2cam rotation (row-major)
        position: 3 floats for camera position
        focal_35mm: Focal length in 35mm equivalent
        pp_u: Normalized principal point U
        pp_v: Normalized principal point V
        distortion_model: RC distortion model name
        distortion: 6 floats [k1, k2, k3, k4, t1, t2]
        
    Returns:
        XMP file content as string
    """
    if distortion is None:
        distortion = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    rotation_str = ' '.join(f'{v:.15g}' for v in rotation)
    position_str = ' '.join(f'{v:.15g}' for v in position)
    distortion_str = ' '.join(f'{v:.15g}' for v in distortion)
    
    return XMP_TEMPLATE.format(
        distortion_model=distortion_model,
        focal_35mm=f'{focal_35mm:.15g}',
        pp_u=f'{pp_u:.15g}',
        pp_v=f'{pp_v:.15g}',
        rotation=rotation_str,
        position=position_str,
        distortion=distortion_str
    )


# =============================================================================
# Main conversion function
# =============================================================================

def convert_sfmdata_to_rc(
    sfmdata_path: str,
    output_folder: str,
    images_folder_name: str = "images",
    copy_images: bool = True,
    sensor_width: float = 36.0
) -> Dict:
    """
    Convert AliceVision SfMData to RealityCapture XMP files.
    
    Creates:
    - output_folder/
      - <images_folder_name>/     # Copied images (if copy_images=True)
        - image1.jpg
        - image2.jpg
        - ...
      - image1.xmp
      - image2.xmp
      - ...
    
    The XMP files contain paths pointing to images in the images folder.
    
    Args:
        sfmdata_path: Path to input SfMData file (.json, .sfm, .abc)
        output_folder: Output folder for XMP files and images
        images_folder_name: Name of the images subfolder (default: "images")
        copy_images: Whether to copy images to output folder
        sensor_width: Sensor width in mm (for focal length conversion)
        
    Returns:
        Dict with conversion statistics
    """
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    images_output_path = output_path / images_folder_name
    if copy_images:
        images_output_path.mkdir(parents=True, exist_ok=True)
    
    # Load SfMData
    sfm = SfMDataWrapper.load(sfmdata_path)
    print(f"Loaded SfMData: {sfmdata_path}")
    
    # Get data as dict for easier access
    sfm_dict = sfm.as_dict() if sfm.has_native else sfm._json_data
    
    # Build lookup dicts
    intrinsics_dict = {}
    for intr in sfm_dict.get('intrinsics', []):
        intrinsics_dict[intr['intrinsicId']] = intr
    
    poses_dict = {}
    for pose in sfm_dict.get('poses', []):
        poses_dict[pose['poseId']] = pose['pose']['transform']
    
    views = sfm_dict.get('views', [])
    print(f"Found {len(views)} views, {len(poses_dict)} poses")
    
    stats = {
        'views_total': len(views),
        'views_processed': 0,
        'views_skipped': 0,
        'images_copied': 0
    }
    
    for view in views:
        view_id = view['viewId']
        intr_id = view['intrinsicId']
        pose_id = view['poseId']
        image_path = view.get('path', '')
        
        # Skip views without pose or intrinsics
        if intr_id not in intrinsics_dict or pose_id not in poses_dict:
            print(f"Warning: Skipping view {view_id} - missing pose or intrinsics")
            stats['views_skipped'] += 1
            continue
        
        intr = intrinsics_dict[intr_id]
        transform = poses_dict[pose_id]
        
        # Get image info
        width = int(intr['width'])
        height = int(intr['height'])
        focal_mm = float(intr['focalLength'])
        intr_sensor_width = float(intr.get('sensorWidth', sensor_width))
        
        # Get principal point offset
        pp = intr.get('principalPoint', ['0', '0'])
        pp_x = float(pp[0])
        pp_y = float(pp[1])
        
        # Get rotation and center
        rotation_flat = [float(r) for r in transform['rotation']]
        R_cam2world = np.array(rotation_flat).reshape(3, 3)
        center = np.array([float(c) for c in transform['center']])
        
        # Get distortion
        dist_type = intr.get('distortionType', 'none')
        dist_params_raw = intr.get('distortionParams', [])
        dist_params = [float(p) for p in dist_params_raw] if dist_params_raw else []
        
        # =====================================================================
        # Apply inverse conversions (AV -> RC)
        # =====================================================================
        
        # Convert rotation
        rotation_rc = av_rotation_to_rc(R_cam2world)
        
        # Convert position
        position_rc = av_position_to_rc(center)
        
        # Convert focal length
        focal_35mm = focal_mm_to_35mm(focal_mm, intr_sensor_width)
        
        # Convert principal point
        pp_u, pp_v = convert_principal_point_to_rc(pp_x, pp_y, width, height)
        
        # Convert distortion
        rc_dist_model, rc_dist_params = convert_distortion_to_rc(dist_type, dist_params)
        
        # =====================================================================
        # Determine output image path and copy if needed
        # =====================================================================
        
        source_image = Path(image_path) if image_path else None
        if source_image and source_image.exists():
            image_filename = source_image.name
            output_image_path = images_output_path / image_filename
            
            if copy_images and not output_image_path.exists():
                shutil.copy2(source_image, output_image_path)
                stats['images_copied'] += 1
        else:
            # Use viewId or generate a name
            image_filename = f"view_{view_id}.jpg"
            output_image_path = images_output_path / image_filename
            if not source_image or not source_image.exists():
                print(f"Warning: Source image not found for view {view_id}: {image_path}")
        
        # =====================================================================
        # Generate XMP file
        # =====================================================================
        
        xmp_content = generate_xmp_content(
            rotation=rotation_rc,
            position=position_rc,
            focal_35mm=focal_35mm,
            pp_u=pp_u,
            pp_v=pp_v,
            distortion_model=rc_dist_model,
            distortion=rc_dist_params
        )
        
        # XMP filename matches image stem
        xmp_filename = Path(image_filename).stem + '.xmp'
        xmp_path = output_path / xmp_filename
        
        with open(xmp_path, 'w', encoding='utf-8') as f:
            f.write(xmp_content)
        
        stats['views_processed'] += 1
    
    print(f"Wrote {stats['views_processed']} XMP files to {output_folder}")
    if copy_images:
        print(f"Copied {stats['images_copied']} images to {images_output_path}")
    
    return stats


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Convert AliceVision SfMData to RealityCapture XMP',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  pyav-sfm2rc sfmdata.json ./output
  pyav-sfm2rc sfmdata.json ./output --images-folder photos
  pyav-sfm2rc sfmdata.json ./output --no-copy-images
  pyav-sfm2rc sfmdata.json ./output --sensor-width 23.5
        """
    )
    parser.add_argument('sfmdata', help='Input SfMData file (.json, .sfm, .abc)')
    parser.add_argument('output_folder', help='Output folder for XMP files and images')
    parser.add_argument(
        '--images-folder', '-i',
        default='images',
        help='Name of the images subfolder (default: images)'
    )
    parser.add_argument(
        '--no-copy-images',
        action='store_true',
        help='Do not copy images to output folder'
    )
    parser.add_argument(
        '--sensor-width',
        type=float,
        default=36.0,
        help='Default sensor width in mm (default: 36.0)'
    )
    
    args = parser.parse_args()
    
    convert_sfmdata_to_rc(
        sfmdata_path=args.sfmdata,
        output_folder=args.output_folder,
        images_folder_name=args.images_folder,
        copy_images=not args.no_copy_images,
        sensor_width=args.sensor_width
    )


if __name__ == '__main__':
    main()
