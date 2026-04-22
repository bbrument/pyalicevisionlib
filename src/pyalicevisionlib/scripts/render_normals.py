#!/usr/bin/env python3
"""
Render normal maps and masks from a mesh and SfMData cameras.

For each camera view, casts rays through every pixel onto the mesh
and produces:
  - A 16-bit PNG normal map in camera space (R=right, G=up, B=towards camera)
  - An 8-bit PNG binary mask
  - A sfm.json copy with image paths replaced by the rendered normal paths

Usage:
    pyav-render-normals <sfmdata> <mesh> -o <output_dir>
"""

import argparse
import json
import sys

import numpy as np
from pathlib import Path
from tqdm import tqdm

from ..sfmdata import load_sfmdata
from ..mesh import load_mesh
from ..rendering import render_normal_map
from ..image import save_image


def main():
    parser = argparse.ArgumentParser(
        description='Render normal maps and masks from mesh and SfMData cameras.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Render all views
  pyav-render-normals sfmdata.json mesh.ply -o output/

  # Render specific views
  pyav-render-normals sfmdata.json mesh.ply -o output/ --views 123456 789012

  # Adjust chunk size for memory
  pyav-render-normals sfmdata.json mesh.ply -o output/ --chunk-size 500000
        """
    )
    parser.add_argument('sfmdata', type=str, help='SfMData file path')
    parser.add_argument('mesh', type=str, help='Mesh file path')
    parser.add_argument('-o', '--output', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--views', nargs='+', type=str, default=None,
                        help='Specific view IDs to render (default: all)')
    parser.add_argument('--chunk-size', type=int, default=1_000_000,
                        help='Rays per batch (default: 1000000)')
    parser.add_argument('--samples', type=int, default=3,
                        help='Sub-pixel grid size for anti-aliasing (default: 3 = 3x3)')

    args = parser.parse_args()

    sfm_path = Path(args.sfmdata)
    mesh_path = Path(args.mesh)

    if not sfm_path.exists():
        print(f"Error: SfMData not found: {sfm_path}")
        sys.exit(1)
    if not mesh_path.exists():
        print(f"Error: Mesh not found: {mesh_path}")
        sys.exit(1)

    output_dir = Path(args.output)
    normals_dir = output_dir / "normals"
    masks_dir = output_dir / "masks"
    normals_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading SfMData: {sfm_path}")
    sfm = load_sfmdata(str(sfm_path))
    cameras = sfm.get_cameras()
    print(f"Loaded {len(cameras)} cameras")

    if args.views is not None:
        view_set = set(args.views)
        cameras = [c for c in cameras if c.view_id in view_set]
        if not cameras:
            print(f"Error: No cameras match view IDs: {args.views}")
            sys.exit(1)
        print(f"Rendering {len(cameras)} selected views")

    print(f"Loading mesh: {mesh_path}")
    tri_mesh = load_mesh(str(mesh_path))
    print(f"Mesh: {len(tri_mesh.vertices)} vertices, {len(tri_mesh.faces)} faces")

    rendered_view_ids = set()
    for camera in tqdm(cameras, desc="Rendering"):
        normal_map, mask = render_normal_map(
            tri_mesh, camera, chunk_size=args.chunk_size, samples=args.samples
        )

        save_image(normal_map, str(normals_dir / f"{camera.view_id}.png"))
        save_image(mask.astype(np.uint8) * 255, str(masks_dir / f"{camera.view_id}.png"))
        rendered_view_ids.add(str(camera.view_id))

    # Save sfm.json with image paths replaced by normal map paths
    sfm_data = sfm.as_dict()
    for view in sfm_data.get('views', []):
        view_id = str(view.get('viewId', ''))
        if view_id in rendered_view_ids:
            view['path'] = str((normals_dir / f"{view_id}.png").resolve())

    sfm_out_path = output_dir / "sfm.json"
    with open(sfm_out_path, 'w') as f:
        json.dump(sfm_data, f, indent=4)

    print(f"Done. Normals: {normals_dir}, Masks: {masks_dir}, SfM: {sfm_out_path}")


if __name__ == '__main__':
    main()
