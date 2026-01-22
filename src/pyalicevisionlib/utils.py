"""
Utility functions for mesh processing, point cloud operations, and transformations.
"""

import numpy as np
import multiprocessing as mp
from pathlib import Path
from typing import Tuple, Optional, Union

# Optional imports
try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False

try:
    import sklearn.neighbors as skln
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


def _sample_single_tri(input_):
    """Sample points from a single triangle based on its subdivision."""
    n1, n2, v1, v2, tri_vert = input_
    c = np.mgrid[:n1+1, :n2+1]
    c += 0.5
    c[0] /= max(n1, 1e-7)
    c[1] /= max(n2, 1e-7)
    c = np.transpose(c, (1, 2, 0))
    k = c[c.sum(axis=-1) < 1]
    q = v1 * k[:, :1] + v2 * k[:, 1:] + tri_vert
    return q


def upsample_mesh_to_pointcloud(
    vertices: np.ndarray,
    triangles: np.ndarray,
    thresh: float = 0.05
) -> np.ndarray:
    """
    Densely sample points from a triangle mesh.
    
    Args:
        vertices: (N, 3) array of vertex coordinates
        triangles: (M, 3) array of triangle indices
        thresh: Sampling threshold controlling point density
        
    Returns:
        (P, 3) array of sampled points including original vertices
    """
    tri_vert = vertices[triangles]
    
    v1 = tri_vert[:, 1] - tri_vert[:, 0]
    v2 = tri_vert[:, 2] - tri_vert[:, 0]
    l1 = np.linalg.norm(v1, axis=-1, keepdims=True)
    l2 = np.linalg.norm(v2, axis=-1, keepdims=True)
    area2 = np.linalg.norm(np.cross(v1, v2), axis=-1, keepdims=True)
    
    non_zero_area = (area2 > 0)[:, 0]
    l1, l2, area2, v1, v2, tri_vert = [
        arr[non_zero_area] for arr in [l1, l2, area2, v1, v2, tri_vert]
    ]
    
    thr = thresh * np.sqrt(l1 * l2 / area2)
    n1 = np.floor(l1 / thr)
    n2 = np.floor(l2 / thr)

    with mp.Pool() as mp_pool:
        new_pts = mp_pool.map(
            _sample_single_tri,
            ((n1[i, 0], n2[i, 0], v1[i:i+1], v2[i:i+1], tri_vert[i:i+1, 0]) 
             for i in range(len(n1))),
            chunksize=1024
        )

    new_pts = np.concatenate(new_pts, axis=0)
    return np.concatenate([vertices, new_pts], axis=0)


def downsample_pointcloud(points: np.ndarray, radius: float) -> np.ndarray:
    """
    Downsample point cloud using radius-based filtering.
    
    Args:
        points: (N, 3) array of points
        radius: Minimum distance between points
        
    Returns:
        Downsampled point cloud
    """
    if not HAS_SKLEARN:
        raise ImportError("scikit-learn is required for downsampling")
    
    nn_engine = skln.NearestNeighbors(
        n_neighbors=1, radius=radius, algorithm='kd_tree', n_jobs=-1
    )
    nn_engine.fit(points)
    rnn_idxs = nn_engine.radius_neighbors(points, radius=radius, return_distance=False)
    
    mask = np.ones(points.shape[0], dtype=np.bool_)
    for curr, idxs in enumerate(rnn_idxs):
        if mask[curr]:
            mask[idxs] = 0
            mask[curr] = 1
    
    return points[mask]


def load_mesh_arrays(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a mesh file and return vertices and triangles as numpy arrays.
    
    Args:
        path: Path to the mesh file (.ply, .obj, etc.)
        
    Returns:
        Tuple of (vertices, triangles) as numpy arrays
    """
    if not HAS_OPEN3D:
        raise ImportError("open3d is required for mesh loading")
    
    mesh = o3d.io.read_triangle_mesh(path)
    vertices = np.asarray(mesh.vertices).astype(np.float32)
    triangles = np.asarray(mesh.triangles).astype(np.int32)
    return vertices, triangles


def save_pointcloud(
    path: str, 
    points: np.ndarray, 
    colors: Optional[np.ndarray] = None
):
    """
    Save a point cloud to a PLY file.
    
    Args:
        path: Output file path
        points: (N, 3) array of point coordinates
        colors: Optional (N, 3) array of RGB colors in [0, 1]
    """
    if not HAS_OPEN3D:
        raise ImportError("open3d is required for saving point clouds")
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(path, pcd)


def remove_mesh_vertices_below_threshold(
    vertices: np.ndarray,
    faces: np.ndarray,
    z_threshold: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove vertices and associated faces below a z-coordinate threshold.
    
    Args:
        vertices: (N, 3) array of vertices
        faces: (M, 3) array of face indices
        z_threshold: Minimum z-coordinate to keep
        
    Returns:
        Tuple of (new_vertices, new_faces)
    """
    mask_keep = vertices[:, 2] >= z_threshold
    return remove_mesh_vertices_by_mask(vertices, faces, mask_keep)


def remove_mesh_vertices_by_mask(
    vertices: np.ndarray,
    faces: np.ndarray,
    keep_mask: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove vertices and associated faces based on a boolean mask.
    
    Args:
        vertices: (N, 3) array of vertices
        faces: (M, 3) array of face indices
        keep_mask: (N,) boolean array, True for vertices to keep
        
    Returns:
        Tuple of (new_vertices, new_faces)
    """
    valid_faces_mask = keep_mask[faces].all(axis=1)
    valid_faces = faces[valid_faces_mask]
    
    shift = np.cumsum(~keep_mask)
    new_faces = valid_faces - shift[valid_faces]
    new_vertices = vertices[keep_mask]
    
    return new_vertices, new_faces


def compute_nearest_neighbor_distances(
    source: np.ndarray,
    target: np.ndarray
) -> np.ndarray:
    """
    Compute distances from each point in source to nearest point in target.
    
    Args:
        source: (N, 3) array of source points
        target: (M, 3) array of target points
        
    Returns:
        (N,) array of distances
    """
    if not HAS_SKLEARN:
        raise ImportError("scikit-learn is required for nearest neighbor computation")
    
    nn_engine = skln.NearestNeighbors(n_neighbors=1, algorithm='kd_tree', n_jobs=-1)
    nn_engine.fit(target)
    distances, _ = nn_engine.kneighbors(source, n_neighbors=1, return_distance=True)
    return distances.flatten()


# =============================================================================
# Transformation utilities
# =============================================================================

def load_transform(path: Union[str, Path]) -> np.ndarray:
    """
    Load a 4x4 transformation matrix from file.
    
    Supports:
    - .npy: NumPy binary format
    - .txt: Text format (whitespace-separated)
    
    Args:
        path: Path to transformation file
        
    Returns:
        4x4 transformation matrix
        
    Raises:
        ValueError: If matrix is not 4x4
    """
    path = Path(path)
    if path.suffix == '.npy':
        T = np.load(path)
    else:
        T = np.loadtxt(path)
    
    if T.shape != (4, 4):
        raise ValueError(f"Transformation must be 4x4, got: {T.shape}")
    
    return T


def decompose_transform(T: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Decompose a 4x4 transformation into rotation, translation, and scale.
    
    Assumes uniform scale (same scale factor along all axes).
    
    Args:
        T: 4x4 transformation matrix
        
    Returns:
        R: 3x3 rotation matrix (pure rotation, no scale)
        t: 3D translation vector
        scale: Uniform scale factor
    """
    R_scaled = T[:3, :3]
    t = T[:3, 3]
    
    # Extract scale (assuming uniform scale)
    scale = np.linalg.norm(R_scaled[:, 0])
    R = R_scaled / scale if scale > 1e-9 else R_scaled
    
    return R, t, scale
