"""
Normal map and mask rendering from mesh and cameras.

Renders per-view normal maps in camera space and binary masks
using ray-mesh intersection (trimesh + embreex backend).

Normal convention (camera space):
    R = right (+X), G = up (+Y), B = towards camera (-Z)

Output encoding:
    normals in [-1, 1] mapped to [0, 65535] uint16
"""

import numpy as np
from typing import Tuple

try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False

from .camera import Camera


def _compute_barycentric(vertices: np.ndarray, points: np.ndarray) -> np.ndarray:
    """
    Compute barycentric coordinates for points inside triangles.

    Args:
        vertices: (N, 3, 3) triangle vertices
        points: (N, 3) hit locations

    Returns:
        (N, 3) barycentric coordinates [w0, w1, w2]
    """
    A = vertices[:, 0]
    B = vertices[:, 1]
    C = vertices[:, 2]

    v0 = B - A
    v1 = C - A
    v2 = points - A

    d00 = np.einsum('ij,ij->i', v0, v0)
    d01 = np.einsum('ij,ij->i', v0, v1)
    d11 = np.einsum('ij,ij->i', v1, v1)
    d20 = np.einsum('ij,ij->i', v2, v0)
    d21 = np.einsum('ij,ij->i', v2, v1)

    denom = d00 * d11 - d01 * d01
    denom = np.where(denom != 0, denom, 1e-8)

    w1 = (d11 * d20 - d01 * d21) / denom
    w2 = (d00 * d21 - d01 * d20) / denom
    w0 = 1.0 - w1 - w2

    return np.column_stack([w0, w1, w2])


def _generate_rays(
    camera: Camera,
    offset_u: float = 0.5,
    offset_v: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate ray origins and directions for all pixels of a camera.

    Args:
        camera: Camera object
        offset_u: Sub-pixel horizontal offset within each pixel (0-1)
        offset_v: Sub-pixel vertical offset within each pixel (0-1)

    Returns:
        Tuple of (origins, directions):
            - origins: (H*W, 3) ray origins (camera center)
            - directions: (H*W, 3) normalized ray directions in world space
    """
    H, W = camera.height, camera.width

    u = np.arange(W, dtype=np.float64) + offset_u
    v = np.arange(H, dtype=np.float64) + offset_v
    uu, vv = np.meshgrid(u, v)

    pixels = np.stack([uu.ravel(), vv.ravel(), np.ones(H * W)], axis=-1)

    K_inv = np.linalg.inv(camera.get_K())
    dirs_cam = (K_inv @ pixels.T).T
    dirs_world = (camera.rotation_cam2world @ dirs_cam.T).T

    norms = np.linalg.norm(dirs_world, axis=1, keepdims=True)
    dirs_world = dirs_world / norms

    origins = np.tile(camera.center, (H * W, 1))

    return origins, dirs_world


def _cast_rays_single(
    mesh: 'trimesh.Trimesh',
    camera: Camera,
    tri_verts: np.ndarray,
    tri_normals: np.ndarray,
    R_w2c: np.ndarray,
    offset_u: float,
    offset_v: float,
    chunk_size: int,
) -> np.ndarray:
    """
    Cast rays for one sub-pixel offset and return per-pixel normals in camera space.

    Args:
        mesh: trimesh.Trimesh object
        camera: Camera object
        tri_verts: (F, 3, 3) triangle vertices
        tri_normals: (F, 3, 3) per-vertex normals for each triangle
        R_w2c: (3, 3) world-to-camera rotation
        offset_u: Sub-pixel horizontal offset (0-1)
        offset_v: Sub-pixel vertical offset (0-1)
        chunk_size: Number of rays per batch

    Returns:
        (H, W, 3) float64 normals in camera space (zeros where no hit)
    """
    H, W = camera.height, camera.width
    origins, directions = _generate_rays(camera, offset_u, offset_v)
    n_rays = len(origins)

    normal_map = np.zeros((H, W, 3), dtype=np.float64)

    for start in range(0, n_rays, chunk_size):
        end = min(start + chunk_size, n_rays)

        hit_faces, hit_rays, hit_locations = mesh.ray.intersects_id(
            ray_origins=origins[start:end],
            ray_directions=directions[start:end],
            return_locations=True,
            multiple_hits=False
        )

        if len(hit_faces) == 0:
            continue

        # Barycentric interpolation of vertex normals
        bary = _compute_barycentric(tri_verts[hit_faces], hit_locations)
        normals_world = np.sum(
            tri_normals[hit_faces] * bary[:, :, np.newaxis], axis=1
        )

        # Normalize
        norms = np.linalg.norm(normals_world, axis=1, keepdims=True)
        normals_world = normals_world / np.where(norms > 0, norms, 1.0)

        # World to camera space
        normals_cam = (R_w2c @ normals_world.T).T

        # Flip Y and Z for normal map convention:
        # Y: camera Y points down in image → flip to "up" positive
        # Z: camera Z points forward → flip to "towards camera" positive
        normals_cam[:, 1] = -normals_cam[:, 1]
        normals_cam[:, 2] = -normals_cam[:, 2]

        global_idx = start + hit_rays
        normal_map[global_idx // W, global_idx % W] = normals_cam

    return normal_map


def render_normal_map(
    mesh: 'trimesh.Trimesh',
    camera: Camera,
    chunk_size: int = 1_000_000,
    samples: int = 3
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Render a normal map and mask for a single camera view.

    Casts rays through all pixels with multi-sample anti-aliasing
    (samples x samples sub-pixel grid, median filter) and computes
    per-pixel normals in camera space using barycentric interpolation
    of vertex normals.

    Normal convention (camera space):
        R = right (+X), G = up (+Y), B = towards camera (-Z)

    Args:
        mesh: trimesh.Trimesh object
        camera: Camera object
        chunk_size: Number of rays per batch (controls memory usage)
        samples: Sub-pixel grid size per axis (3 = 3x3 = 9 samples, 1 = no AA)

    Returns:
        Tuple of (normal_map, mask):
            - normal_map: (H, W, 3) uint16 in [0, 65535]
            - mask: (H, W) bool array (True where mesh is visible)
    """
    if not HAS_TRIMESH:
        raise ImportError("trimesh is required for rendering")

    H, W = camera.height, camera.width

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    vertex_normals = np.asarray(mesh.vertex_normals, dtype=np.float64)
    tri_verts = vertices[faces]
    tri_normals = vertex_normals[faces]
    R_w2c = camera.rotation_world2cam

    # Sub-pixel offsets (e.g. samples=3 -> [0.25, 0.5, 0.75])
    offsets = np.linspace(0, 1, samples + 2)[1:-1]
    n_samples = len(offsets) ** 2

    # Accumulate samples: (H, W, 3, n_samples)
    all_samples = np.zeros((H, W, 3, n_samples), dtype=np.float64)
    idx = 0
    for ou in offsets:
        for ov in offsets:
            all_samples[:, :, :, idx] = _cast_rays_single(
                mesh, camera, tri_verts, tri_normals, R_w2c, ou, ov, chunk_size
            )
            idx += 1

    # Median across samples then re-normalize
    normal_map = np.median(all_samples, axis=3)
    norms = np.linalg.norm(normal_map, axis=2, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    normal_map = normal_map / norms

    # Mask: at least one sample hit
    mask = np.any(
        np.linalg.norm(all_samples, axis=2) > 0, axis=2
    )

    # Zero out normals where no sample hit (keep the zero vector)
    normal_map[~mask] = 0.0

    # [-1, 1] -> [0, 65535] uint16
    # Background (0,0,0) encodes to midpoint (32767,32767,32767) = gray
    normal_map_uint16 = ((normal_map + 1.0) / 2.0 * 65535).clip(0, 65535).astype(np.uint16)

    return normal_map_uint16, mask
