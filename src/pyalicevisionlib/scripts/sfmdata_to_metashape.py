#!/usr/bin/env python3
"""
Convert AliceVision SfMData to Agisoft Metashape XML.

AliceVision SfMData format:
- Rotation: cam2world (row-major)
- Center: camera center in world
- PrincipalPoint: offset from image center in pixels
- Distortion: radialk3 [k1, k2, k3] or brown [k1, k2, k3, p1, p2]

Metashape XML format:
- Camera transforms: 4x4 cam2world matrices (row-major)
- Component transform: identity (rotation=I, translation=0, scale=1)
- Sensor calibration: f (pixels), cx/cy (pixel offset from center), k1-k3, p1-p2

Usage:
    pyav-sfm2metashape <sfmdata.json> <output.xml> [--sensor-width 36.0]
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List
from xml.sax.saxutils import escape as xml_escape

import numpy as np

from ..sfmdata import SfMDataWrapper, WORLD_CORRECTION_4x4

logger = logging.getLogger(__name__)


# =============================================================================
# XML Templates
# =============================================================================

DOCUMENT_TEMPLATE = '''<?xml version="1.0" encoding="UTF-8"?>
<document version="2.0.0">
  <chunk label="Chunk 1" enabled="true">
    <sensors next_id="{next_sensor_id}">{sensors_xml}
    </sensors>
    <components next_id="1" active_id="0">
      <component id="0" label="Component 1">
        <transform>
          <rotation>1 0 0 0 1 0 0 0 1</rotation>
          <translation>0 0 0</translation>
          <scale>1</scale>
        </transform>
        <partition>
          <camera_ids>{camera_ids}</camera_ids>
        </partition>
      </component>
    </components>
    <cameras next_id="{next_camera_id}" next_group_id="0">{cameras_xml}
    </cameras>
  </chunk>
</document>
'''

SENSOR_TEMPLATE = '''
      <sensor id="{id}" label="{label}" type="frame">
        <resolution width="{w}" height="{h}"/>
        <calibration type="frame" class="adjusted">
          <resolution width="{w}" height="{h}"/>
          <f>{focal_px}</f>
          <cx>{cx}</cx>
          <cy>{cy}</cy>
          <k1>{k1}</k1>
          <k2>{k2}</k2>
          <k3>{k3}</k3>
          <p1>{p1}</p1>
          <p2>{p2}</p2>
        </calibration>
      </sensor>'''

CAMERA_TEMPLATE = '''
      <camera id="{id}" sensor_id="{sensor_id}" component_id="0" label="{label}">
        <transform>{transform}</transform>
      </camera>'''


# =============================================================================
# Conversion functions
# =============================================================================

def extract_distortion_params(intrinsic: Dict) -> Dict[str, float]:
    """Extract distortion parameters from SfMData intrinsic.

    Handles 'radialk3' (3 params: k1,k2,k3), 'brown' (5 params: k1,k2,k3,p1,p2),
    and 'none' (all zeros).

    Returns:
        Dict with keys k1, k2, k3, p1, p2.
    """
    dist_type = intrinsic.get('distortionType', 'none')
    params = intrinsic.get('distortionParams', [])
    params = [float(p) for p in params] if params else []

    if dist_type == 'none' or not params:
        return {'k1': 0.0, 'k2': 0.0, 'k3': 0.0, 'p1': 0.0, 'p2': 0.0}
    elif dist_type == 'radialk3':
        k1 = params[0] if len(params) > 0 else 0.0
        k2 = params[1] if len(params) > 1 else 0.0
        k3 = params[2] if len(params) > 2 else 0.0
        return {'k1': k1, 'k2': k2, 'k3': k3, 'p1': 0.0, 'p2': 0.0}
    elif dist_type == 'brown':
        k1 = params[0] if len(params) > 0 else 0.0
        k2 = params[1] if len(params) > 1 else 0.0
        k3 = params[2] if len(params) > 2 else 0.0
        p1 = params[3] if len(params) > 3 else 0.0
        p2 = params[4] if len(params) > 4 else 0.0
        return {'k1': k1, 'k2': k2, 'k3': k3, 'p1': p1, 'p2': p2}
    else:
        # Unknown type, try to extract what we can
        k1 = params[0] if len(params) > 0 else 0.0
        k2 = params[1] if len(params) > 1 else 0.0
        k3 = params[2] if len(params) > 2 else 0.0
        return {'k1': k1, 'k2': k2, 'k3': k3, 'p1': 0.0, 'p2': 0.0}


def convert_sfmdata_to_metashape(
    sfmdata_path: str,
    output_path: str,
    sensor_width: float = 36.0,
) -> Dict:
    """Convert AliceVision SfMData to Metashape XML.

    Args:
        sfmdata_path: Path to input SfMData file
        output_path: Output Metashape XML path
        sensor_width: Sensor width in mm (for focal px conversion)

    Returns:
        Dict with conversion statistics
    """
    sfm = SfMDataWrapper.load(sfmdata_path)
    sfm_dict = sfm.as_dict() if sfm.has_native else sfm._json_data

    # Build lookup dicts
    intrinsics_by_id = {}
    for intr in sfm_dict.get('intrinsics', []):
        intrinsics_by_id[intr['intrinsicId']] = intr

    poses_by_id = {}
    for pose in sfm_dict.get('poses', []):
        poses_by_id[pose['poseId']] = pose['pose']['transform']

    # Map intrinsicIds to sequential sensor ids
    intrinsic_ids = sorted(intrinsics_by_id.keys())
    intrinsic_to_sensor = {iid: str(idx) for idx, iid in enumerate(intrinsic_ids)}

    # Build sensor XML
    sensors_xml_parts = []
    for iid in intrinsic_ids:
        intr = intrinsics_by_id[iid]
        sid = intrinsic_to_sensor[iid]
        w = int(intr['width'])
        h = int(intr['height'])
        focal_mm = float(intr['focalLength'])
        intr_sensor_width = float(intr.get('sensorWidth', sensor_width))

        # Convert focal_mm -> focal_px
        focal_px = focal_mm * w / intr_sensor_width

        # Principal point (same convention, pass through)
        pp = intr.get('principalPoint', ['0', '0'])
        cx = float(pp[0])
        cy = float(pp[1])

        # Distortion
        dist = extract_distortion_params(intr)

        sensors_xml_parts.append(SENSOR_TEMPLATE.format(
            id=sid,
            label=intr.get('serialNumber', f'Sensor {sid}'),
            w=w,
            h=h,
            focal_px=f'{focal_px:.15g}',
            cx=f'{cx:.15g}',
            cy=f'{cy:.15g}',
            k1=f'{dist["k1"]:.15g}',
            k2=f'{dist["k2"]:.15g}',
            k3=f'{dist["k3"]:.15g}',
            p1=f'{dist["p1"]:.15g}',
            p2=f'{dist["p2"]:.15g}',
        ))

    # Build camera XML
    cameras_xml_parts = []
    camera_id_list = []
    cam_idx = 0

    views = sfm_dict.get('views', [])
    for view in views:
        pose_id = view['poseId']
        intr_id = view['intrinsicId']

        if pose_id not in poses_by_id or intr_id not in intrinsic_to_sensor:
            continue

        transform = poses_by_id[pose_id]
        sensor_id = intrinsic_to_sensor[intr_id]

        # Get rotation and center from SfMData
        rotation_flat = [float(r) for r in transform['rotation']]
        R = np.array(rotation_flat).reshape(3, 3)
        center = np.array([float(c) for c in transform['center']])

        # Build AV 4x4 pose [R | center; 0 0 0 1]
        av_pose = np.eye(4)
        av_pose[:3, :3] = R
        av_pose[:3, 3] = center

        # Undo world correction: WORLD_CORRECTION_4x4 is its own inverse
        metashape_pose = WORLD_CORRECTION_4x4 @ av_pose

        # Since component transform is identity, camera local = world
        transform_str = ' '.join(f'{v:.15g}' for v in metashape_pose.flatten())

        # Camera label = image filename stem
        image_path = view.get('path', '')
        label = Path(image_path).stem if image_path else f'camera_{cam_idx}'

        cameras_xml_parts.append(CAMERA_TEMPLATE.format(
            id=cam_idx,
            sensor_id=sensor_id,
            label=xml_escape(label),
            transform=transform_str,
        ))
        camera_id_list.append(str(cam_idx))
        cam_idx += 1

    # Assemble full XML
    xml_content = DOCUMENT_TEMPLATE.format(
        next_sensor_id=len(intrinsic_ids),
        sensors_xml=''.join(sensors_xml_parts),
        camera_ids=' '.join(camera_id_list),
        next_camera_id=cam_idx,
        cameras_xml=''.join(cameras_xml_parts),
    )

    # Write output
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, 'w', encoding='utf-8') as f:
        f.write(xml_content)

    logger.info("Converted %d cameras to Metashape XML", cam_idx)
    logger.info("  Sensors: %d", len(intrinsic_ids))
    logger.info("  Output: %s", output_path)

    return {
        'cameras': cam_idx,
        'sensors': len(intrinsic_ids),
    }


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Convert AliceVision SfMData to Agisoft Metashape XML',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  pyav-sfm2metashape sfmdata.json metashape.xml
  pyav-sfm2metashape sfmdata.json metashape.xml --sensor-width 23.5
        """,
    )
    parser.add_argument('sfmdata', help='Input SfMData file (.json, .sfm, .abc)')
    parser.add_argument('output', help='Output Metashape XML file')
    parser.add_argument(
        '--sensor-width',
        type=float,
        default=36.0,
        help='Sensor width in mm (default: 36.0)',
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    convert_sfmdata_to_metashape(
        args.sfmdata,
        args.output,
        args.sensor_width,
    )


if __name__ == '__main__':
    main()
