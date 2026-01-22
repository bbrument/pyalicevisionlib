"""
Chamfer distance computation for mesh evaluation.

Provides metrics for comparing reconstructed meshes against ground truth,
including Chamfer distance, precision, recall, and F-score at various thresholds.
"""

import numpy as np
import csv
import os
from typing import Tuple, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt

from ..utils import (
    load_mesh_arrays,
    upsample_mesh_to_pointcloud,
    downsample_pointcloud,
    compute_nearest_neighbor_distances,
    save_pointcloud
)


@dataclass
class ChamferResult:
    """Results from Chamfer distance computation."""
    chamfer_data2gt: float
    chamfer_gt2data: float
    chamfer_mean: float
    n_data_points: int
    n_gt_points: int
    
    def to_dict(self) -> dict:
        return {
            'chamfer_data2gt': self.chamfer_data2gt,
            'chamfer_gt2data': self.chamfer_gt2data,
            'chamfer_mean': self.chamfer_mean,
            'n_data_points': self.n_data_points,
            'n_gt_points': self.n_gt_points
        }


@dataclass  
class PrecisionRecallResult:
    """Precision-recall results at multiple thresholds."""
    thresholds: np.ndarray
    precisions: np.ndarray
    recalls: np.ndarray
    fscores: np.ndarray
    
    def get_at_threshold(self, threshold: float) -> Tuple[float, float, float]:
        """Get precision, recall, f-score at a specific threshold."""
        idx = np.argmin(np.abs(self.thresholds - threshold))
        return self.precisions[idx], self.recalls[idx], self.fscores[idx]


def mesh_to_pointcloud(
    mesh_path: str,
    sampling_density: float = 0.05,
    cache_path: Optional[str] = None,
    use_open3d: bool = True
) -> np.ndarray:
    """
    Convert a mesh to a densely sampled point cloud.
    
    Args:
        mesh_path: Path to the mesh file
        sampling_density: Point sampling density (distance between points)
        cache_path: Optional path to cache the sampled point cloud
        use_open3d: If True, use Open3D for fast sampling (recommended for large meshes)
        
    Returns:
        (N, 3) array of sampled points
    """
    import open3d as o3d
    
    # Check cache
    if cache_path and os.path.exists(cache_path):
        pcd = o3d.io.read_point_cloud(cache_path)
        return np.asarray(pcd.points)
    
    if use_open3d:
        # Use Open3D for fast sampling (much faster for large meshes)
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        
        if not mesh.has_triangles():
            # It's a point cloud, not a mesh
            pcd = o3d.io.read_point_cloud(mesh_path)
            points = np.asarray(pcd.points)
        else:
            # Estimate number of points based on surface area and density
            mesh.compute_triangle_normals()
            surface_area = mesh.get_surface_area()
            
            # Number of points = surface_area / (density^2)
            # density is the average distance between points
            n_points = int(surface_area / (sampling_density ** 2))
            n_points = max(n_points, 10000)  # At least 10k points
            
            # Sample points uniformly
            pcd = mesh.sample_points_uniformly(number_of_points=n_points)
            points = np.asarray(pcd.points)
        
        # Remove NaNs
        points = points[~np.isnan(points).any(axis=1)]
        
        # Cache if requested
        if cache_path:
            save_pointcloud(cache_path, points)
        
        return points
    
    # Fallback: use custom upsampling (slower but works without Open3D mesh support)
    vertices, triangles = load_mesh_arrays(mesh_path)
    points = upsample_mesh_to_pointcloud(vertices, triangles, sampling_density)
    
    # Remove NaNs
    points = points[~np.isnan(points).any(axis=1)]
    
    # Shuffle for better downsampling
    np.random.shuffle(points)
    
    # Downsample
    points = downsample_pointcloud(points, sampling_density)
    
    # Cache if requested
    if cache_path:
        save_pointcloud(cache_path, points)
    
    return points


def compute_chamfer_distance(
    data_points: np.ndarray,
    gt_points: np.ndarray,
    max_dist: float = float('inf')
) -> ChamferResult:
    """
    Compute Chamfer distance between two point clouds.
    
    Args:
        data_points: (N, 3) array of reconstructed points
        gt_points: (M, 3) array of ground truth points
        max_dist: Maximum distance for outlier filtering
        
    Returns:
        ChamferResult with distance metrics
    """
    # Compute distances
    dist_data2gt = compute_nearest_neighbor_distances(data_points, gt_points)
    dist_gt2data = compute_nearest_neighbor_distances(gt_points, data_points)
    
    # Filter outliers
    dist_data2gt_filtered = dist_data2gt[dist_data2gt < max_dist]
    dist_gt2data_filtered = dist_gt2data[dist_gt2data < max_dist]
    
    # Compute mean distances
    chamfer_data2gt = np.mean(dist_data2gt_filtered)
    chamfer_gt2data = np.mean(dist_gt2data_filtered)
    chamfer_mean = (chamfer_data2gt + chamfer_gt2data) / 2
    
    return ChamferResult(
        chamfer_data2gt=chamfer_data2gt,
        chamfer_gt2data=chamfer_gt2data,
        chamfer_mean=chamfer_mean,
        n_data_points=len(data_points),
        n_gt_points=len(gt_points)
    )


def compute_precision_recall(
    data_points: np.ndarray,
    gt_points: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
    max_dist: float = float('inf')
) -> PrecisionRecallResult:
    """
    Compute precision, recall, and F-score at multiple distance thresholds.
    
    - Precision: fraction of reconstructed points within threshold of GT
    - Recall: fraction of GT points within threshold of reconstruction
    - F-score: harmonic mean of precision and recall
    
    Args:
        data_points: (N, 3) array of reconstructed points
        gt_points: (M, 3) array of ground truth points
        thresholds: Array of distance thresholds to evaluate
        max_dist: Maximum distance for outlier filtering
        
    Returns:
        PrecisionRecallResult with metrics at each threshold
    """
    if thresholds is None:
        thresholds = np.linspace(0.001, 1.5, 1000)
    
    # Compute distances
    dist_data2gt = compute_nearest_neighbor_distances(data_points, gt_points)
    dist_gt2data = compute_nearest_neighbor_distances(gt_points, data_points)
    
    # Filter outliers
    dist_data2gt_filtered = dist_data2gt[dist_data2gt < max_dist]
    dist_gt2data_filtered = dist_gt2data[dist_gt2data < max_dist]
    
    n_data = len(dist_data2gt_filtered)
    n_gt = len(dist_gt2data_filtered)
    
    precisions = np.zeros(len(thresholds))
    recalls = np.zeros(len(thresholds))
    fscores = np.zeros(len(thresholds))
    
    for i, thresh in enumerate(thresholds):
        precision = np.sum(dist_data2gt_filtered < thresh) / n_data if n_data > 0 else 0
        recall = np.sum(dist_gt2data_filtered < thresh) / n_gt if n_gt > 0 else 0
        
        if precision + recall > 0:
            fscore = 2 * precision * recall / (precision + recall)
        else:
            fscore = 0
            
        precisions[i] = precision
        recalls[i] = recall
        fscores[i] = fscore
    
    return PrecisionRecallResult(
        thresholds=thresholds,
        precisions=precisions,
        recalls=recalls,
        fscores=fscores
    )


def evaluate_mesh(
    data_mesh_path: str,
    gt_mesh_path: str,
    output_dir: str,
    sampling_density: float = 0.05,
    max_dist: float = 2.0,
    thresholds: Optional[np.ndarray] = None,
    visualize: bool = False,
    vis_threshold: float = 1.0,
    z_min_filter: Optional[float] = None,
    verbose: bool = True
) -> Tuple[ChamferResult, PrecisionRecallResult]:
    """
    Full mesh evaluation pipeline: Chamfer distance + precision/recall.
    
    Args:
        data_mesh_path: Path to reconstructed mesh
        gt_mesh_path: Path to ground truth mesh
        output_dir: Directory for output files
        sampling_density: Point cloud sampling density
        max_dist: Maximum distance for outlier filtering
        thresholds: Distance thresholds for precision/recall
        visualize: Generate visualization point clouds
        vis_threshold: Threshold for error visualization colormap
        z_min_filter: Optional minimum z-coordinate filter
        verbose: Print progress messages
        
    Returns:
        Tuple of (ChamferResult, PrecisionRecallResult)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Sample ground truth point cloud
    if verbose:
        print("Sampling ground truth point cloud...")
    gt_cache = os.path.join(os.path.dirname(gt_mesh_path), 'gt_pcd.ply')
    gt_points = mesh_to_pointcloud(gt_mesh_path, sampling_density, gt_cache)
    gt_z_min = np.min(gt_points[:, 2])
    
    if verbose:
        print(f"Ground truth: {len(gt_points)} points")
    
    # Sample data point cloud
    if verbose:
        print("Sampling reconstructed mesh point cloud...")
    data_points = mesh_to_pointcloud(data_mesh_path, sampling_density)
    
    # Filter by z-coordinate
    if z_min_filter is not None:
        z_filter = z_min_filter
    else:
        z_filter = gt_z_min
    
    data_points = data_points[data_points[:, 2] > z_filter]
    
    if verbose:
        print(f"Reconstructed: {len(data_points)} points")
    
    # Compute Chamfer distance
    if verbose:
        print("Computing Chamfer distance...")
    chamfer = compute_chamfer_distance(data_points, gt_points, max_dist)
    
    if verbose:
        print(f"Chamfer distance: {chamfer.chamfer_mean:.4f}")
        print(f"  - Data to GT: {chamfer.chamfer_data2gt:.4f}")
        print(f"  - GT to Data: {chamfer.chamfer_gt2data:.4f}")
    
    # Save Chamfer results
    chamfer_csv = os.path.join(output_dir, 'chamfer_distance.csv')
    with open(chamfer_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['data2gt', 'gt2data', 'mean'])
        writer.writerow([chamfer.chamfer_data2gt, chamfer.chamfer_gt2data, chamfer.chamfer_mean])
    
    # Compute precision/recall
    if verbose:
        print("Computing precision/recall/F-score...")
    pr_result = compute_precision_recall(data_points, gt_points, thresholds, max_dist)
    
    # Save precision/recall results
    pr_csv = os.path.join(output_dir, 'precision_recall.csv')
    with open(pr_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(pr_result.thresholds)
        writer.writerow(pr_result.precisions)
        writer.writerow(pr_result.recalls)
        writer.writerow(pr_result.fscores)
    
    # Plot F-score curve
    if verbose:
        print("Generating F-score plot...")
    plt.figure(figsize=(10, 6))
    plt.plot(pr_result.thresholds, pr_result.fscores)
    plt.xlabel('Distance threshold')
    plt.ylabel('F-score')
    plt.title('F-score vs Distance Threshold')
    plt.xlim([0, pr_result.thresholds.max()])
    plt.ylim([0, 1])
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'fscore_curve.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Generate visualizations
    if visualize:
        if verbose:
            print("Generating error visualization...")
        
        dist_data2gt = compute_nearest_neighbor_distances(data_points, gt_points)
        dist_gt2data = compute_nearest_neighbor_distances(gt_points, data_points)
        
        # Color map for errors
        data_alpha = np.clip(dist_data2gt / vis_threshold, 0, 1)
        data_colors = plt.cm.jet(data_alpha)[:, :3]
        save_pointcloud(os.path.join(output_dir, 'vis_data2gt.ply'), data_points, data_colors)
        
        gt_alpha = np.clip(dist_gt2data / vis_threshold, 0, 1)
        gt_colors = plt.cm.jet(gt_alpha)[:, :3]
        save_pointcloud(os.path.join(output_dir, 'vis_gt2data.ply'), gt_points, gt_colors)
    
    if verbose:
        print(f"Results saved to {output_dir}")
    
    return chamfer, pr_result
