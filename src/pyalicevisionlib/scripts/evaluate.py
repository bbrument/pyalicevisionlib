#!/usr/bin/env python3
"""
Evaluate mesh reconstruction quality against ground truth.

Usage:
    pyav-evaluate --data-mesh mesh.ply --gt-mesh gt.ply --output-dir results/
    
Or:
    python -m pyalicevisionlib.scripts.evaluate --data-mesh mesh.ply --gt-mesh gt.ply
"""

import argparse
import os
import sys

from ..evaluation import run_evaluation


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate mesh reconstruction quality against ground truth",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic evaluation (no cleanup)
  pyav-evaluate --data-mesh mesh.ply --gt-mesh gt.ply --output-dir results/

  # With visibility-based cleanup (masks from alpha channel in sfmData images)
  pyav-evaluate --data-mesh mesh.ply --gt-mesh gt.ply \\
      --cameras-masks sfm.json --output-dir results/ --cleanup

  # Full options
  pyav-evaluate --data-mesh mesh.ply --gt-mesh gt.ply --cameras-masks sfm.json \\
      --output-dir results/ --cleanup --z-threshold 0.0 \\
      --sampling-density 0.03 --visualize
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--data-mesh", required=True,
        help="Path to reconstructed mesh (.ply, .obj, etc.)"
    )
    parser.add_argument(
        "--gt-mesh", required=True,
        help="Path to ground truth mesh"
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Directory for output files"
    )
    
    # Optional arguments
    parser.add_argument(
        "--cameras-masks",
        help="Path to AliceVision sfmData JSON with mask images (PNG with alpha or mask files). Required for --cleanup."
    )
    parser.add_argument(
        "--cleanup", action="store_true",
        help="Perform visibility-based mesh cleanup before evaluation"
    )
    parser.add_argument(
        "--dilation-radius", type=int, default=12,
        help="Radius for mask dilation in pixels (default: 12)"
    )
    parser.add_argument(
        "--z-threshold", type=float, default=None,
        help="Minimum z-coordinate threshold for filtering"
    )
    parser.add_argument(
        "--sampling-density", type=float, default=0.05,
        help="Point cloud sampling density (default: 0.05)"
    )
    parser.add_argument(
        "--max-dist", type=float, default=2.0,
        help="Maximum distance for outlier filtering (default: 2.0)"
    )
    parser.add_argument(
        "--visualize", action="store_true",
        help="Generate error visualization point clouds"
    )
    parser.add_argument(
        "--vis-threshold", type=float, default=1.0,
        help="Threshold for error visualization colormap (default: 1.0)"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress progress messages"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.exists(args.data_mesh):
        print(f"Error: Data mesh not found: {args.data_mesh}")
        sys.exit(1)
    
    if not os.path.exists(args.gt_mesh):
        print(f"Error: Ground truth mesh not found: {args.gt_mesh}")
        sys.exit(1)
    
    if args.cleanup and args.cameras_masks is None:
        print("Error: --cameras-masks is required when --cleanup is enabled")
        sys.exit(1)
    
    if args.cameras_masks and not os.path.exists(args.cameras_masks):
        print(f"Error: Cameras/masks file not found: {args.cameras_masks}")
        sys.exit(1)
    
    # Run evaluation
    run_evaluation(
        data_mesh_path=args.data_mesh,
        gt_mesh_path=args.gt_mesh,
        output_dir=args.output_dir,
        cameras_path=args.cameras_masks,
        cleanup=args.cleanup,
        use_alpha=True,  # Always use alpha channel from images in sfmData
        dilation_radius=args.dilation_radius,
        z_threshold=args.z_threshold,
        sampling_density=args.sampling_density,
        max_dist=args.max_dist,
        visualize=args.visualize,
        vis_threshold=args.vis_threshold,
        verbose=not args.quiet
    )


if __name__ == "__main__":
    main()
