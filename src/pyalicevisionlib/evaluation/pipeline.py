"""
Main evaluation pipeline for mesh reconstruction quality.

Combines mesh cleanup and Chamfer distance evaluation.
"""

import os
import json
from datetime import datetime
from typing import Optional

from .chamfer import evaluate_mesh, ChamferResult, PrecisionRecallResult
from .cleanup import cleanup_mesh_visibility


def run_evaluation(
    data_mesh_path: str,
    gt_mesh_path: str,
    output_dir: str,
    cameras_path: Optional[str] = None,
    cleanup: bool = False,
    use_alpha: bool = True,
    dilation_radius: int = 12,
    z_threshold: Optional[float] = None,
    sampling_density: float = 0.05,
    max_dist: float = 2.0,
    visualize: bool = False,
    vis_threshold: float = 1.0,
    verbose: bool = True
) -> dict:
    """
    Run complete mesh evaluation pipeline.
    
    Args:
        data_mesh_path: Path to reconstructed mesh
        gt_mesh_path: Path to ground truth mesh
        output_dir: Directory for output files
        cameras_path: Path to AliceVision sfmData JSON (required for cleanup).
                     Image paths in sfmData should point to mask images (PNG with alpha channel).
        cleanup: Whether to perform visibility-based mesh cleanup
        use_alpha: If True, use alpha channel from images as masks (default: True)
        dilation_radius: Radius for mask dilation in pixels (default: 12)
        z_threshold: Optional minimum z-coordinate threshold
        sampling_density: Point cloud sampling density
        max_dist: Maximum distance for outlier filtering
        visualize: Generate visualization point clouds
        vis_threshold: Threshold for error visualization colormap
        verbose: Print progress messages
        
    Returns:
        Dictionary with evaluation results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Track the mesh to evaluate
    mesh_to_evaluate = data_mesh_path
    
    # Step 1: Optional cleanup
    if cleanup:
        if cameras_path is None:
            raise ValueError("cameras_path is required when cleanup is enabled")
        
        if verbose:
            print("=" * 60)
            print("Step 1: Mesh Cleanup (visibility-based)")
            print("=" * 60)
        
        cleaned_mesh_path = os.path.join(output_dir, "mesh_cleaned.ply")
        cleanup_mesh_visibility(
            mesh_path=data_mesh_path,
            cameras_path=cameras_path,
            output_path=cleaned_mesh_path,
            use_alpha=use_alpha,
            dilation_radius=dilation_radius,
            z_threshold=z_threshold,
            verbose=verbose
        )
        mesh_to_evaluate = cleaned_mesh_path
        
        if verbose:
            print()
    
    # Step 2: Chamfer distance evaluation
    if verbose:
        print("=" * 60)
        print("Step 2: Chamfer Distance Evaluation")
        print("=" * 60)
    
    chamfer, pr_result = evaluate_mesh(
        data_mesh_path=mesh_to_evaluate,
        gt_mesh_path=gt_mesh_path,
        output_dir=output_dir,
        sampling_density=sampling_density,
        max_dist=max_dist,
        visualize=visualize,
        vis_threshold=vis_threshold,
        verbose=verbose
    )
    
    # Step 3: Generate summary report
    if verbose:
        print()
        print("=" * 60)
        print("Evaluation Summary")
        print("=" * 60)
    
    # Get F-score at common thresholds
    common_thresholds = [0.5, 1.0, 1.5]
    fscore_at_thresholds = {}
    for t in common_thresholds:
        p, r, f = pr_result.get_at_threshold(t)
        fscore_at_thresholds[f"f{t}"] = f
        if verbose:
            print(f"  F-score @ {t}: {f:.4f} (P={p:.4f}, R={r:.4f})")
    
    if verbose:
        print()
        print(f"Chamfer Distance:")
        print(f"  - Data → GT:  {chamfer.chamfer_data2gt:.4f}")
        print(f"  - GT → Data:  {chamfer.chamfer_gt2data:.4f}")
        print(f"  - Mean:       {chamfer.chamfer_mean:.4f}")
    
    # Build results dictionary
    results = {
        "timestamp": datetime.now().isoformat(),
        "input": {
            "data_mesh": data_mesh_path,
            "gt_mesh": gt_mesh_path,
            "cameras": cameras_path,
            "cleanup": cleanup,
            "z_threshold": z_threshold,
            "sampling_density": sampling_density,
            "max_dist": max_dist
        },
        "chamfer": chamfer.to_dict(),
        "fscore_at_thresholds": fscore_at_thresholds,
        "output_dir": output_dir
    }
    
    # Save summary JSON
    summary_path = os.path.join(output_dir, "evaluation_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    if verbose:
        print()
        print(f"Results saved to: {output_dir}")
        print(f"Summary: {summary_path}")
    
    return results
