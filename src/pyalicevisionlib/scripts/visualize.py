#!/usr/bin/env python3
"""
Visualize cameras and meshes from SfMData files.

Usage:
    pyav-visualize <sfmdata.json> [--mesh <mesh.ply>] [--mesh-points 5000]
    
Or:
    python -m pyalicevisionlib.scripts.visualize <sfmdata.json>
"""

import argparse
import numpy as np
from pathlib import Path
from typing import List, Optional

from ..sfmdata import load_cameras_from_sfmdata, load_sfmdata
from ..mesh import sample_mesh_points
from ..camera import Camera
from ..visualization import visualize_cameras, compare_cameras, print_intrinsics


def main():
    parser = argparse.ArgumentParser(
        description='Visualize cameras and meshes from SfMData',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single SfMData file
  pyav-visualize sfmdata.json --mesh mesh.ply --mesh-points 5000

  # Compare multiple camera sets
  pyav-visualize cameras1.json cameras2.json --labels "Set 1" "Set 2"

  # Print intrinsics info
  pyav-visualize sfmdata.json --print-intrinsics
        """
    )
    parser.add_argument('sfmdata_files', nargs='+', help='One or more SfMData files')
    parser.add_argument('--labels', nargs='+', help='Labels for each file')
    parser.add_argument('--mesh', type=str, help='Mesh file to display')
    parser.add_argument('--mesh-points', type=int, default=5000, help='Max mesh points to display')
    parser.add_argument('--axis-scale', type=float, default=0.1, help='Camera axis scale')
    parser.add_argument('--no-axes', action='store_true', help='Hide camera coordinate axes')
    parser.add_argument('--print-intrinsics', action='store_true', help='Print intrinsics info')
    parser.add_argument('--title', default='Camera Visualization', help='Plot title')
    
    args = parser.parse_args()
    
    # Load mesh
    mesh_points = None
    if args.mesh:
        mesh_points = sample_mesh_points(args.mesh, args.mesh_points)
        print(f"Loaded {len(mesh_points)} mesh points from {args.mesh}")
    
    # Load cameras
    cameras_list = []
    labels = args.labels or [Path(f).stem for f in args.sfmdata_files]
    
    for sfm_file, label in zip(args.sfmdata_files, labels):
        sfm = load_sfmdata(sfm_file)
        cameras = sfm.get_cameras()
        cameras_list.append(cameras)
        print(f"Loaded {len(cameras)} cameras from {label}")
        
        if args.print_intrinsics:
            print_intrinsics(cameras, label)
    
    # Visualize
    if len(cameras_list) == 1:
        visualize_cameras(
            cameras_list[0],
            mesh_points=mesh_points,
            axis_scale=args.axis_scale,
            show_axes=not args.no_axes,
            title=args.title,
            label=labels[0]
        )
    else:
        compare_cameras(
            cameras_list,
            labels,
            mesh_points=mesh_points,
            axis_scale=args.axis_scale,
            title=args.title
        )


if __name__ == '__main__':
    main()
