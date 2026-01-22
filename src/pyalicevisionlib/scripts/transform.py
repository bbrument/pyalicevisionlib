#!/usr/bin/env python3
"""
Apply a 4x4 transformation to mesh and/or camera poses.

This script transforms meshes and SfMData camera poses consistently.

Usage:
    # Transform mesh only
    pyav-transform --mesh input.ply -t transform.npy -o output.ply
    
    # Transform mesh and cameras
    pyav-transform --mesh input.ply -t transform.npy -o output.ply \\
                   --cameras sfm.json --cameras-output sfm_transformed.json
    
    # Apply inverse transformation
    pyav-transform --mesh input.ply -t transform.npy -o output.ply --inverse
    
    # Transform cameras only (no mesh)
    pyav-transform -t transform.npy --cameras sfm.json --cameras-output sfm_transformed.json

Convention:
    The transformation T is applied to the WORLD coordinates.
    
    For the mesh:
        v' = T @ v
    
    For camera poses, the mathematically correct formula is used to preserve
    projection invariance:
        M' = S @ T @ S @ M
    
    Where S = diag([1, -1, -1, 1]) is the world correction matrix and
    M is the 4x4 cam2world pose matrix.
    
    This ensures that: P' @ S @ X' = P @ S @ X
    i.e., the projection of transformed points equals the projection of
    original points (same pixel coordinates).
"""

import argparse
import numpy as np
from pathlib import Path
from typing import Tuple

from ..sfmdata import load_sfmdata
from ..utils import load_transform, decompose_transform


def apply_transform_to_mesh(mesh_path: Path, T: np.ndarray, output_path: Path):
    """
    Apply transformation to mesh and save result.
    
    Args:
        mesh_path: Input mesh path
        T: 4x4 transformation matrix
        output_path: Output mesh path
    """
    import trimesh
    
    mesh = trimesh.load_mesh(mesh_path)
    print(f"Loaded mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    
    mesh_transformed = mesh.copy()
    mesh_transformed.apply_transform(T)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mesh_transformed.export(output_path)
    print(f"Saved transformed mesh: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Apply 4x4 transformation to mesh and/or camera poses.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Transform mesh only
    pyav-transform --mesh model.ply -t transform.npy -o model_aligned.ply
    
    # Transform mesh and cameras
    pyav-transform --mesh model.ply -t transform.npy -o model_aligned.ply \\
                   --cameras sfm.json --cameras-output sfm_aligned.json
    
    # Apply inverse transformation
    pyav-transform --mesh model.ply -t transform.npy -o model_aligned.ply --inverse
        """
    )
    parser.add_argument("--mesh", "-m", type=str, default=None,
                       help="Input mesh path (.ply, .obj, etc.)")
    parser.add_argument("--transform", "-t", type=str, required=True,
                       help="4x4 transformation matrix (.npy or .txt)")
    parser.add_argument("--output", "-o", type=str, default=None,
                       help="Output mesh path")
    parser.add_argument("--inverse", "-i", action='store_true',
                       help="Apply inverse transformation")
    parser.add_argument("--cameras", "-c", type=str, default=None,
                       help="Input SfMData JSON path")
    parser.add_argument("--cameras-output", type=str, default=None,
                       help="Output SfMData JSON path")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.mesh is None and args.cameras is None:
        parser.error("At least one of --mesh or --cameras is required")
    
    if args.mesh is not None and args.output is None:
        parser.error("--output is required when --mesh is specified")
    
    if args.cameras is not None and args.cameras_output is None:
        parser.error("--cameras-output is required when --cameras is specified")
    
    # Load transformation
    transform_path = Path(args.transform)
    print(f"Loading transformation: {transform_path}")
    T = load_transform(transform_path)
    
    if args.inverse:
        T = np.linalg.inv(T)
        print("Using INVERSE transformation")
    
    print("Transformation matrix:")
    print(T)
    
    R_T, t_T, scale = decompose_transform(T)
    print(f"Scale: {scale:.6f}")
    print(f"Translation: {t_T}")
    
    # Transform mesh
    if args.mesh is not None:
        mesh_path = Path(args.mesh)
        output_path = Path(args.output)
        print(f"\nTransforming mesh: {mesh_path}")
        apply_transform_to_mesh(mesh_path, T, output_path)
    
    # Transform cameras using the wrapper's apply_transform method
    # This uses the mathematically correct formula: M' = S @ T @ S @ M
    if args.cameras is not None:
        cameras_path = Path(args.cameras)
        cameras_output_path = Path(args.cameras_output)
        
        print(f"\nTransforming cameras: {cameras_path}")
        sfm_wrapper = load_sfmdata(str(cameras_path))
        
        sfmdata_orig = sfm_wrapper.as_dict()
        num_poses = len(sfmdata_orig.get('poses', []))
        print(f"Found {num_poses} poses")
        print(f"Using formula: M' = S @ T @ S @ M (preserves projection invariance)")
        
        # Apply transformation using the wrapper method
        sfm_transformed = sfm_wrapper.apply_transform(T)
        
        # Save transformed SfMData
        sfm_transformed.save(str(cameras_output_path))
        print(f"Saved transformed cameras: {cameras_output_path}")
        
        # Verification summary
        print("\n--- Verification (first 3 cameras) ---")
        sfmdata_new = sfm_transformed.as_dict()
        poses_orig = sfmdata_orig.get('poses', [])[:3]
        poses_new = sfmdata_new.get('poses', [])[:3]
        
        for orig, new in zip(poses_orig, poses_new):
            pose_id = orig.get('poseId', '?')
            c_orig = np.array([float(x) for x in orig['pose']['transform']['center']])
            c_new = np.array([float(x) for x in new['pose']['transform']['center']])
            
            r_orig = np.array([float(x) for x in orig['pose']['transform']['rotation']]).reshape(3, 3)
            r_new = np.array([float(x) for x in new['pose']['transform']['rotation']]).reshape(3, 3)
            
            # Look direction is 3rd column of cam2world
            look_orig = r_orig[:, 2]
            look_new = r_new[:, 2]
            
            print(f"Pose {pose_id}:")
            print(f"  Center: {c_orig} -> {c_new}")
            print(f"  Look dir: {look_orig} -> {look_new}")


if __name__ == "__main__":
    main()