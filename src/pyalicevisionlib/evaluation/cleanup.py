"""
Mesh cleanup based on camera visibility and 2D masks.

This module provides functionality to clean a mesh by removing faces
that are not visible from any camera viewpoint. Optionally uses 2D
mask images (silhouettes) to further filter points.

Based on meshCleanupVisibility.py reference implementation.
"""

import os
import numpy as np
from typing import List, Optional, Tuple, Union
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

# Fallback to skimage if cv2 not available
try:
    from skimage.morphology import binary_dilation
    from skimage.draw import disk
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

from ..camera import Camera
from ..sfmdata import load_sfmdata
from ..image import load_mask, load_alpha_as_mask


def _create_dilation_kernel(radius: int) -> np.ndarray:
    """
    Create a circular dilation kernel.
    
    Args:
        radius: Radius of the circular kernel
        
    Returns:
        2D binary/uint8 array with circular disk
    """
    if HAS_CV2:
        # cv2 uses diameter, returns uint8
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))
    elif HAS_SKIMAGE:
        kernel_size = 2 * radius - 1
        kernel = np.zeros((kernel_size, kernel_size), dtype=bool)
        rr, cc = disk((radius - 1, radius - 1), radius, shape=kernel.shape)
        kernel[rr, cc] = True
        return kernel
    else:
        raise ImportError("cv2 or scikit-image is required for mask dilation")


def _dilate_mask(mask: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Dilate a binary mask with a kernel.
    
    Uses cv2 (37x faster) if available, falls back to skimage.
    
    Args:
        mask: Binary mask to dilate
        kernel: Dilation kernel
        
    Returns:
        Dilated binary mask (bool array)
    """
    if HAS_CV2:
        # cv2.dilate requires uint8
        mask_uint8 = mask.astype(np.uint8) * 255 if mask.dtype == bool else mask
        dilated = cv2.dilate(mask_uint8, kernel)
        return dilated > 0
    elif HAS_SKIMAGE:
        return binary_dilation(mask, kernel)
    else:
        raise ImportError("cv2 or scikit-image is required for mask dilation")


def _load_and_dilate_mask(
    source_path: str,
    kernel: np.ndarray,
    use_alpha: bool = False
) -> np.ndarray:
    """
    Load and dilate a single mask (for parallel processing).
    
    Args:
        source_path: Path to mask or image file
        kernel: Dilation kernel
        use_alpha: If True, extract mask from alpha channel
        
    Returns:
        Dilated binary mask
    """
    if use_alpha:
        mask = load_alpha_as_mask(source_path)
    else:
        mask = load_mask(source_path)
    return _dilate_mask(mask, kernel)


def _load_masks_parallel(
    source_paths: List[str],
    kernel: np.ndarray,
    use_alpha: bool = False,
    max_workers: int = 8,
    verbose: bool = True
) -> List[np.ndarray]:
    """
    Load and dilate all masks in parallel.
    
    Args:
        source_paths: List of paths to mask or image files
        kernel: Dilation kernel
        use_alpha: If True, extract mask from alpha channel
        max_workers: Maximum number of parallel workers
        verbose: Show progress bar
        
    Returns:
        List of dilated binary masks
    """
    n_masks = len(source_paths)
    masks = [None] * n_masks
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_idx = {
            executor.submit(_load_and_dilate_mask, path, kernel, use_alpha): i
            for i, path in enumerate(source_paths)
        }
        
        # Collect results with progress bar
        iterator = as_completed(future_to_idx)
        if verbose:
            iterator = tqdm(iterator, total=n_masks, desc="Loading masks", leave=False)
        
        for future in iterator:
            idx = future_to_idx[future]
            masks[idx] = future.result()
    
    return masks


def _project_points_to_image(
    points: np.ndarray,
    camera: Camera
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project 3D points to image coordinates.
    
    Args:
        points: (N, 3) array of 3D points
        camera: Camera object
        
    Returns:
        Tuple of (projected_2d, valid_mask):
            - projected_2d: (N, 2) image coordinates
            - valid_mask: (N,) boolean array, True if in front of camera
    """
    # Transform to camera coordinates
    points_cam = camera.world_to_camera(points)
    in_front = points_cam[:, 2] > 0
    
    # Project to 2D
    projected = camera.project_points(points)
    
    return projected, in_front


def _check_points_in_bounds(
    projected: np.ndarray,
    width: int,
    height: int
) -> np.ndarray:
    """
    Check if projected points are within image bounds.
    
    Args:
        projected: (N, 2) array of 2D coordinates
        width: Image width
        height: Image height
        
    Returns:
        (N,) boolean array
    """
    return (
        (projected[:, 0] >= 0) &
        (projected[:, 0] < width) &
        (projected[:, 1] >= 0) &
        (projected[:, 1] < height)
    )


def _check_points_in_mask(
    projected: np.ndarray,
    mask: np.ndarray
) -> np.ndarray:
    """
    Check if projected points fall inside a binary mask.
    
    Args:
        projected: (N, 2) array of integer 2D coordinates
        mask: (H, W) binary mask
        
    Returns:
        (N,) boolean array
    """
    h, w = mask.shape
    # Ensure coordinates are valid integers
    x = np.clip(projected[:, 0].astype(int), 0, w - 1)
    y = np.clip(projected[:, 1].astype(int), 0, h - 1)
    return mask[y, x]


def _remove_vertices_by_mask(
    mesh: 'trimesh.Trimesh',
    keep_mask: np.ndarray
) -> 'trimesh.Trimesh':
    """
    Remove vertices (and their faces) based on a boolean mask.
    
    Args:
        mesh: Input trimesh
        keep_mask: Boolean array, True = keep vertex
        
    Returns:
        New trimesh with filtered vertices
    """
    # Find valid faces (all vertices must be kept)
    valid_faces = keep_mask[mesh.faces].all(axis=1)
    kept_faces = mesh.faces[valid_faces]
    
    # Reindex vertices
    shift = np.cumsum(~keep_mask)
    new_faces = kept_faces - shift[kept_faces]
    
    # Create new mesh
    new_mesh = trimesh.Trimesh(
        vertices=mesh.vertices[keep_mask],
        faces=new_faces
    )
    new_mesh.update_faces(new_mesh.nondegenerate_faces())
    
    return new_mesh


def cleanup_mesh_with_masks(
    mesh_path: str,
    cameras: List[Camera],
    output_path: str,
    mask_paths: Optional[List[str]] = None,
    use_alpha: bool = False,
    image_paths: Optional[List[str]] = None,
    dilation_radius: int = 12,
    z_threshold: Optional[float] = None,
    ray_offset_factor: float = 0.05 / 100,
    verbose: bool = True
) -> 'trimesh.Trimesh':
    """
    Clean a mesh using camera visibility and optional 2D masks.
    
    The cleanup process:
    1. Remove vertices below z_threshold (optional)
    2. Remove vertices outside all camera frustums
    3. Remove vertices outside all dilated masks (if provided)
    4. Ray-cast to keep only vertices visible from at least one camera
    
    Args:
        mesh_path: Path to input mesh
        cameras: List of Camera objects
        output_path: Path to save cleaned mesh
        mask_paths: Optional list of mask image paths (one per camera)
        use_alpha: If True, use alpha channel from image_paths as masks
        image_paths: Image paths to extract alpha channel from (required if use_alpha=True)
        dilation_radius: Radius for mask dilation (default: 12)
        z_threshold: Optional minimum z-coordinate filter
        ray_offset_factor: Factor for ray origin offset (default: 0.01/100)
        verbose: Show progress messages
        
    Returns:
        Cleaned trimesh object
    """
    if not HAS_TRIMESH:
        raise ImportError("trimesh is required for mesh cleanup")
    
    n_views = len(cameras)
    
    # Determine mask source
    use_masks = False
    mask_source = None  # Will be either mask_paths or image_paths
    
    if mask_paths is not None and len(mask_paths) == n_views:
        use_masks = True
        mask_source = mask_paths
        mask_mode = "mask_files"
    elif use_alpha and image_paths is not None and len(image_paths) == n_views:
        use_masks = True
        mask_source = image_paths
        mask_mode = "alpha_channel"
    
    if use_masks and not HAS_SKIMAGE:
        raise ImportError("scikit-image is required for mask-based cleanup")
    
    # Create progress bar
    if verbose:
        total_steps = 6 if use_masks else 5
        pbar = tqdm(total=total_steps, desc="Cleanup")
    
    # Load mesh
    if verbose:
        pbar.set_description("Loading mesh")
    mesh = trimesh.load(mesh_path)
    mesh.update_faces(mesh.nondegenerate_faces())
    
    if verbose:
        print(f"\nInitial mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
        if use_masks:
            print(f"Mask mode: {mask_mode}")
    
    # Step 1: Z-threshold filter
    if z_threshold is not None:
        if verbose:
            pbar.update(1)
            pbar.set_description("Z-threshold filter")
        
        keep_mask = mesh.vertices[:, 2] >= z_threshold
        mesh = _remove_vertices_by_mask(mesh, keep_mask)
        
        if verbose:
            print(f"After z-threshold: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    
    # Create dilation kernel if using masks
    if use_masks:
        dilation_kernel = _create_dilation_kernel(dilation_radius)
        
        # Pre-load and dilate all masks in parallel
        if verbose:
            pbar.update(1)
            pbar.set_description("Loading masks")
        
        use_alpha_mode = (mask_mode == "alpha_channel")
        dilated_masks = _load_masks_parallel(
            mask_source, dilation_kernel, use_alpha=use_alpha_mode, verbose=verbose
        )
    
    # Step 2: Remove vertices outside all camera frustums
    if verbose:
        pbar.update(1)
        pbar.set_description("Frustum culling")
    
    vertices = mesh.vertices.astype(np.float32)
    n_points = len(vertices)
    inside_any_frustum = np.zeros(n_points, dtype=bool)
    
    for camera in cameras:
        projected, in_front = _project_points_to_image(vertices, camera)
        in_bounds = _check_points_in_bounds(projected, camera.width, camera.height)
        inside_any_frustum |= (in_front & in_bounds)
    
    mesh = _remove_vertices_by_mask(mesh, inside_any_frustum)
    
    if verbose:
        print(f"After frustum culling: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    
    # Step 3: Remove vertices outside all masks (if using masks)
    if use_masks:
        if verbose:
            pbar.update(1)
            pbar.set_description("Mask filtering")
        
        for i, camera in enumerate(cameras):
            dilated_mask = dilated_masks[i]
            
            # Project vertices
            vertices = mesh.vertices.astype(np.float32)
            n_points = len(vertices)
            projected, in_front = _project_points_to_image(vertices, camera)
            
            h, w = dilated_mask.shape
            in_bounds = _check_points_in_bounds(projected, w, h)
            
            # Check which points are in the mask
            in_mask = np.zeros(n_points, dtype=bool)
            valid_for_mask = in_front & in_bounds
            if valid_for_mask.any():
                valid_projected = projected[valid_for_mask].astype(int)
                in_mask_subset = _check_points_in_mask(valid_projected, dilated_mask)
                in_mask[valid_for_mask] = in_mask_subset
            
            # Keep points that are either outside frustum OR inside mask
            keep = ~valid_for_mask | in_mask
            mesh = _remove_vertices_by_mask(mesh, keep)
        
        if verbose:
            print(f"After mask filtering: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    
    # Step 4: Ray-cast visibility check
    if verbose:
        pbar.update(1)
        pbar.set_description("Ray casting")
    
    vertices = mesh.vertices.astype(np.float32)
    n_points = len(vertices)
    
    # Compute ray directions and offset
    ray_norms_max = 0
    for camera in cameras:
        ray_dirs = camera.center - vertices
        ray_norms_max = max(ray_norms_max, np.linalg.norm(ray_dirs, axis=-1).max())
    
    ray_offset = ray_offset_factor * ray_norms_max
    
    # Track visibility per view
    hits = np.zeros((n_views, n_points), dtype=bool)
    in_mask_all_views = np.ones((n_views, n_points), dtype=bool) if use_masks else None
    
    iterator = tqdm(range(n_views), desc="Ray casting", leave=False) if verbose else range(n_views)
    
    for i in iterator:
        camera = cameras[i]
        
        # Compute rays from vertices to camera center
        ray_directions = camera.center - vertices
        ray_norms = np.linalg.norm(ray_directions, axis=-1, keepdims=True)
        ray_directions_norm = ray_directions / ray_norms
        ray_origins = vertices + ray_offset * ray_directions_norm
        
        # If using masks, check mask visibility (using pre-loaded masks)
        if use_masks:
            dilated_mask = dilated_masks[i]
            
            projected, in_front = _project_points_to_image(vertices, camera)
            h, w = dilated_mask.shape
            in_bounds = _check_points_in_bounds(projected, w, h)
            
            in_mask = np.zeros(n_points, dtype=bool)
            valid = in_front & in_bounds
            if valid.any():
                valid_proj = projected[valid].astype(int)
                in_mask[valid] = _check_points_in_mask(valid_proj, dilated_mask)
            
            in_mask_all_views[i] = in_mask
        
        # Ray-mesh intersection
        hit = mesh.ray.intersects_any(
            ray_origins=ray_origins,
            ray_directions=ray_directions_norm
        )
        hits[i] = hit
    
    # Keep points visible in at least one view
    if use_masks:
        visible = ~hits & in_mask_all_views
    else:
        visible = ~hits
    
    visible_in_any_view = visible.any(axis=0)
    mesh = _remove_vertices_by_mask(mesh, visible_in_any_view)
    
    if verbose:
        pbar.update(1)
        print(f"After visibility check: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    
    # Step 5: Export
    if verbose:
        pbar.update(1)
        pbar.set_description("Exporting")
    
    # Create output directory if needed
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    mesh.export(output_path)
    
    if verbose:
        pbar.close()
        print(f"Saved cleaned mesh to {output_path}")
    
    return mesh


def cleanup_mesh_visibility(
    mesh_path: str,
    cameras_path: str,
    output_path: str,
    mask_dir: Optional[str] = None,
    use_alpha: bool = False,
    dilation_radius: int = 12,
    z_threshold: Optional[float] = None,
    apply_world_correction: bool = True,
    verbose: bool = True
) -> 'trimesh.Trimesh':
    """
    Clean a mesh by removing faces not visible from any camera.
    
    This is the main entry point for mesh cleanup. It loads cameras from
    an SfMData file and optionally uses 2D masks for additional filtering.
    
    Args:
        mesh_path: Path to input mesh file
        cameras_path: Path to AliceVision SfMData file (.sfm, .json, .abc)
        output_path: Path to save cleaned mesh
        mask_dir: Optional directory containing mask images (one per view)
        use_alpha: If True, use alpha channel from SfMData images as masks
        dilation_radius: Radius for mask dilation (default: 12 pixels)
        z_threshold: Optional minimum z-coordinate threshold
        apply_world_correction: Whether to apply Y/Z flip to camera poses.
            Default: True (standard Meshroom behavior).
        verbose: Show progress messages
        
    Returns:
        Cleaned trimesh object
    """
    if not HAS_TRIMESH:
        raise ImportError("trimesh is required for mesh cleanup")
    
    if verbose:
        print(f"Loading cameras from {cameras_path}")
    
    # Load cameras from SfMData
    sfm = load_sfmdata(cameras_path)
    cameras = sfm.get_cameras(apply_world_correction=apply_world_correction)
    
    if verbose:
        print(f"Loaded {len(cameras)} cameras")
    
    # Determine mask source
    mask_paths = None
    image_paths = None
    
    if mask_dir is not None and os.path.isdir(mask_dir):
        # Use mask files from directory
        mask_files = sorted(os.listdir(mask_dir))
        if len(mask_files) == len(cameras):
            mask_paths = [os.path.join(mask_dir, f) for f in mask_files]
            if verbose:
                print(f"Found {len(mask_paths)} mask images from directory")
        else:
            if verbose:
                print(f"Warning: {len(mask_files)} masks found but {len(cameras)} cameras. Skipping masks.")
    elif use_alpha:
        # Use alpha channel from SfMData images
        view_to_path = sfm.get_viewid_to_image_path()
        # Get image paths in same order as cameras
        image_paths = []
        for cam in cameras:
            if cam.view_id in view_to_path:
                image_paths.append(view_to_path[cam.view_id])
            else:
                # Fallback: use image_path from camera
                image_paths.append(cam.image_path)
        
        if verbose:
            print(f"Using alpha channel from {len(image_paths)} images as masks")
    
    return cleanup_mesh_with_masks(
        mesh_path=mesh_path,
        cameras=cameras,
        output_path=output_path,
        mask_paths=mask_paths,
        use_alpha=use_alpha,
        image_paths=image_paths,
        dilation_radius=dilation_radius,
        z_threshold=z_threshold,
        verbose=verbose
    )
