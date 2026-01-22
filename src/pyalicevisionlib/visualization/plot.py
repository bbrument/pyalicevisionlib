"""
3D visualization of cameras and meshes.

Provides matplotlib-based visualization tools.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Optional

from ..camera import Camera


def visualize_cameras(
    cameras: List[Camera],
    mesh_points: Optional[np.ndarray] = None,
    axis_scale: float = 0.1,
    show_axes: bool = True,
    title: str = 'Camera Visualization',
    label: str = 'Cameras',
    figsize: tuple = (14, 10)
):
    """
    Visualize cameras in 3D using matplotlib.
    
    Args:
        cameras: List of Camera objects
        mesh_points: Optional (N, 3) array of mesh vertices to display
        axis_scale: Scale for camera axis visualization
        show_axes: Whether to show camera coordinate axes
        title: Plot title
        label: Label for camera points in legend
        figsize: Figure size tuple (width, height)
    
    Note:
        If cameras appear inverted relative to the mesh, load cameras with
        `apply_world_correction=True` in `load_cameras_from_sfmdata()`.
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Draw mesh points if provided
    if mesh_points is not None and len(mesh_points) > 0:
        ax.scatter(
            mesh_points[:, 0], mesh_points[:, 1], mesh_points[:, 2],
            c='gray', s=1, alpha=0.3, label='Mesh'
        )
    
    # Draw cameras
    centers = []
    for cam in cameras:
        center = cam.center
        centers.append(center)
        
        # Camera position
        ax.scatter(*center, c='blue', s=30, alpha=0.8)
        
        if show_axes:
            # Camera axes (columns of cam2world)
            R = cam.rotation_cam2world
            x_axis = R[:, 0] * axis_scale
            y_axis = R[:, 1] * axis_scale
            z_axis = R[:, 2] * axis_scale
            
            ax.quiver(*center, *x_axis, color='red', alpha=0.5, arrow_length_ratio=0.1)
            ax.quiver(*center, *y_axis, color='green', alpha=0.5, arrow_length_ratio=0.1)
            ax.quiver(*center, *z_axis, color='blue', alpha=0.5, arrow_length_ratio=0.1)
        
        # Look direction
        look_dir = cam.get_look_direction() * axis_scale * 2
        ax.quiver(*center, *look_dir, color='orange', alpha=0.8, arrow_length_ratio=0.15, linewidth=2)
    
    # Set equal aspect ratio
    centers = np.array(centers)
    if len(centers) > 0:
        all_points = centers
        if mesh_points is not None and len(mesh_points) > 0:
            all_points = np.vstack([centers, mesh_points])
        
        max_range = np.max(np.ptp(all_points, axis=0)) / 2
        mid = np.mean(all_points, axis=0)
        
        ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
        ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
        ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
    
    # Legend entry
    ax.scatter([], [], [], c='blue', s=50, label=label)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.legend()
    
    plt.tight_layout()
    plt.show()


def compare_cameras(
    cameras_list: List[List[Camera]],
    labels: List[str],
    mesh_points: Optional[np.ndarray] = None,
    axis_scale: float = 0.1,
    title: str = 'Camera Comparison',
    figsize: tuple = (14, 10)
):
    """
    Compare multiple camera sets in 3D.
    
    Args:
        cameras_list: List of camera lists to compare
        labels: Labels for each camera set
        mesh_points: Optional mesh vertices
        axis_scale: Scale for camera axes
        title: Plot title
        figsize: Figure size tuple (width, height)
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(cameras_list)))
    
    # Draw mesh
    if mesh_points is not None and len(mesh_points) > 0:
        ax.scatter(
            mesh_points[:, 0], mesh_points[:, 1], mesh_points[:, 2],
            c='gray', s=1, alpha=0.3, label='Mesh'
        )
    
    all_centers = []
    
    for cameras, label, color in zip(cameras_list, labels, colors):
        centers = []
        for cam in cameras:
            center = cam.center
            centers.append(center)
            
            ax.scatter(*center, c=[color], s=30, alpha=0.8)
            
            look_dir = cam.get_look_direction() * axis_scale * 2
            ax.quiver(*center, *look_dir, color=color, alpha=0.8, arrow_length_ratio=0.15, linewidth=2)
        
        centers = np.array(centers)
        all_centers.append(centers)
        ax.scatter([], [], [], c=[color], s=50, label=label)
    
    # Set equal aspect ratio
    all_centers = np.vstack(all_centers)
    all_points = all_centers
    if mesh_points is not None and len(mesh_points) > 0:
        all_points = np.vstack([all_centers, mesh_points])
    
    max_range = np.max(np.ptp(all_points, axis=0)) / 2
    mid = np.mean(all_points, axis=0)
    
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.legend()
    
    plt.tight_layout()
    plt.show()


def print_intrinsics(cameras: List[Camera], label: str = ""):
    """
    Print intrinsics summary for a list of cameras.
    
    Args:
        cameras: List of Camera objects
        label: Optional label for the camera set
    """
    if not cameras:
        return
    
    print(f"\n{'='*60}")
    print(f"INTRINSICS: {label}")
    print(f"{'='*60}")
    
    cam = cameras[0]
    print(f"  Image size: {cam.width} x {cam.height}")
    print(f"  Focal length: {cam.focal_length_mm:.4f} mm ({cam.focal_length_pixels:.2f} px)")
    print(f"  Sensor width: {cam.sensor_width:.4f} mm")
    print(f"  Principal point offset: ({cam.principal_point[0]:.2f}, {cam.principal_point[1]:.2f}) px")
    print(f"  Principal point absolute: ({cam.cx:.2f}, {cam.cy:.2f}) px")
    print(f"  Number of cameras: {len(cameras)}")
