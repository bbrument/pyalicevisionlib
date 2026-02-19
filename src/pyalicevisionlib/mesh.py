"""
Mesh loading and processing utilities.
"""

import numpy as np
from typing import Union

# Optional trimesh import
try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False

# Optional open3d import
try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False


def load_mesh(filepath: str) -> 'trimesh.Trimesh':
    """
    Load a mesh file using trimesh, with open3d fallback.

    Handles both single meshes and scenes with multiple geometries.

    Args:
        filepath: Path to mesh file (PLY, OBJ, STL, GLTF, etc.)

    Returns:
        trimesh.Trimesh object

    Raises:
        ImportError: If neither trimesh nor open3d is installed
        ValueError: If no valid mesh geometry found
    """
    if not HAS_TRIMESH and not HAS_OPEN3D:
        raise ImportError("trimesh or open3d is required for mesh loading.")

    if HAS_TRIMESH:
        try:
            mesh = trimesh.load(filepath)
            if isinstance(mesh, trimesh.Scene):
                meshes = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
                if meshes:
                    mesh = trimesh.util.concatenate(meshes)
                else:
                    raise ValueError(f"No valid mesh geometry found in {filepath}")
            return mesh
        except (ValueError, Exception) as e:
            if not HAS_OPEN3D:
                raise
            print(f"trimesh failed ({e}), falling back to open3d")

    # Open3d fallback
    o3d_mesh = o3d.io.read_triangle_mesh(filepath)
    vertices = np.asarray(o3d_mesh.vertices)
    faces = np.asarray(o3d_mesh.triangles)
    if len(vertices) == 0:
        raise ValueError(f"No valid mesh geometry found in {filepath}")
    return trimesh.Trimesh(vertices=vertices, faces=faces) if HAS_TRIMESH else o3d_mesh


def sample_mesh_points(
    filepath: str,
    max_points: int = 5000,
    uniform: bool = True
) -> np.ndarray:
    """
    Load mesh and return sampled points.
    
    Args:
        filepath: Path to mesh file
        max_points: Maximum number of points to return
        uniform: If True, sample uniformly on surface; if False, use vertices
        
    Returns:
        (N, 3) array of point positions
    """
    mesh = load_mesh(filepath)
    
    if uniform and hasattr(mesh, 'sample'):
        # Sample uniformly on mesh surface
        points, _ = trimesh.sample.sample_surface(mesh, max_points)
    else:
        # Use mesh vertices
        points = np.array(mesh.vertices)
        if len(points) > max_points:
            indices = np.random.choice(len(points), max_points, replace=False)
            points = points[indices]
    
    return np.array(points)
