#!/usr/bin/env python3
"""
Generate Canny edge contours from images referenced in SfMData.

This script extracts Canny edge contours filtered by a margin around masks.

Algorithm:
1. Load mask from alpha channel or separate mask file
2. Create margin around mask (dilate - erode)
3. Apply Canny edge detection on grayscale image
4. Keep only edges within the margin

Usage:
    pyav-contours --sfm sfmdata.json --masks masks_folder/ --output contours/
    pyav-contours --sfm sfmdata.json --output contours/ --use-alpha
"""

import argparse
import copy
import cv2
import json
import multiprocessing
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
from tqdm import tqdm

from ..sfmdata import load_sfmdata, load_sfmdata_json, save_sfmdata_json
from ..image import load_gray, load_mask, load_alpha_as_mask


def load_image_gray(image_path: str) -> np.ndarray:
    """
    Load image and convert to 8-bit grayscale.
    
    Supports EXR, HDR, and standard image formats via unified image module.
    """
    return load_gray(image_path, dtype='uint8')


def load_mask_from_alpha(image_path: str) -> Optional[np.ndarray]:
    """Load mask from alpha channel of an image."""
    try:
        mask = load_alpha_as_mask(image_path)
        # Convert boolean to uint8 (0/255) for cv2 operations
        return mask.astype(np.uint8) * 255
    except ValueError:
        return None


def load_mask_from_file(mask_path: str, use_alpha: bool = False) -> np.ndarray:
    """Load mask from file (grayscale or alpha channel)."""
    if use_alpha:
        mask = load_mask_from_alpha(mask_path)
        if mask is not None:
            return mask
    
    # Load as binary mask (bool) and convert to uint8
    mask = load_mask(mask_path)
    return mask.astype(np.uint8) * 255


def filter_connected_components(mask: np.ndarray, mode: str = 'hybrid') -> np.ndarray:
    """
    Filter connected components and keep only one.
    
    Modes:
    - 'center_point': Component containing image center
    - 'smallest_area': Smallest component
    - 'hybrid': center_point with smallest_area fallback
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    if num_labels <= 2:  # Background + 0 or 1 component
        return mask
    
    h, w = mask.shape
    center_x, center_y = w // 2, h // 2
    
    if mode == 'center_point' or mode == 'hybrid':
        center_label = labels[center_y, center_x]
        if center_label > 0:
            filtered = np.zeros_like(mask)
            filtered[labels == center_label] = 255
            return filtered
    
    if mode == 'smallest_area' or mode == 'hybrid':
        areas = [stats[i, cv2.CC_STAT_AREA] for i in range(1, num_labels)]
        smallest_idx = np.argmin(areas) + 1
        filtered = np.zeros_like(mask)
        filtered[labels == smallest_idx] = 255
        return filtered
    
    return mask


def extract_canny_contours(
    image_gray: np.ndarray,
    mask: np.ndarray,
    margin_size: int = 20,
    canny_low: int = 50,
    canny_high: int = 150,
    sobel_kernel: int = 3
) -> np.ndarray:
    """
    Extract Canny contours within a margin around the mask.
    
    Args:
        image_gray: Grayscale image
        mask: Binary mask (object region)
        margin_size: Margin width in pixels
        canny_low: Canny low threshold
        canny_high: Canny high threshold
        sobel_kernel: Sobel kernel size (3, 5, or 7)
    
    Returns:
        Binary image with Canny edges within the margin
    """
    # Create margin (dilate - erode)
    kernel = np.ones((margin_size * 2 + 1, margin_size * 2 + 1), np.uint8)
    dilated = cv2.dilate(mask, kernel, iterations=1)
    eroded = cv2.erode(mask, kernel, iterations=1)
    margin = cv2.subtract(dilated, eroded)
    
    # Apply Canny on full image
    edges = cv2.Canny(image_gray, canny_low, canny_high, apertureSize=sobel_kernel)
    
    # Keep only edges within margin
    contours = cv2.bitwise_and(edges, margin)
    return contours


def process_single_image(args: Tuple) -> Dict:
    """Worker function for processing a single image."""
    view_id, image_path, mask_path, config = args
    
    try:
        # Load grayscale image
        gray = load_image_gray(image_path)
        
        # Load mask
        if config['use_alpha']:
            mask = load_mask_from_alpha(mask_path)
            if mask is None:
                return {'status': 'skipped', 'view_id': view_id, 'message': 'No alpha channel'}
        else:
            mask = load_mask_from_file(mask_path)
        
        # Filter connected components
        mask = filter_connected_components(mask, mode=config['component_mode'])
        
        # Extract contours
        contours = extract_canny_contours(
            gray, mask,
            margin_size=config['margin_size'],
            canny_low=config['canny_low'],
            canny_high=config['canny_high'],
            sobel_kernel=config['sobel_kernel']
        )
        
        # Save output
        output_file = config['output_folder'] / f"{view_id}.png"
        cv2.imwrite(str(output_file), contours)
        
        return {'status': 'success', 'view_id': view_id, 'output_path': str(output_file)}
    
    except Exception as e:
        return {'status': 'error', 'view_id': view_id, 'message': str(e)}


def process_contours(
    sfm_path: str,
    masks_folder: str,
    output_folder: str,
    use_alpha: bool = False,
    margin_size: int = 20,
    canny_low: int = 50,
    canny_high: int = 150,
    sobel_kernel: int = 3,
    component_mode: str = 'hybrid',
    num_workers: Optional[int] = None,
    force: bool = False
):
    """
    Process all images and generate Canny contours.
    
    Args:
        sfm_path: Path to SfMData JSON file
        masks_folder: Path to folder containing masks
        output_folder: Output folder for contour images
        use_alpha: Extract mask from alpha channel of mask files
        margin_size: Margin width in pixels
        canny_low: Canny low threshold
        canny_high: Canny high threshold
        sobel_kernel: Sobel kernel size
        component_mode: Connected component filtering mode
        num_workers: Number of parallel workers
        force: Regenerate existing outputs
    """
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    masks_path = Path(masks_folder) if masks_folder else None
    
    # Load SfMData
    sfm = load_sfmdata(sfm_path)
    view_mapping = sfm.get_viewid_to_image_path()
    print(f"Found {len(view_mapping)} views in SfMData")
    
    # Find mask files
    mask_files = {}
    if masks_path:
        for ext in ['*.png', '*.exr', '*.tif', '*.tiff']:
            for f in masks_path.glob(ext):
                mask_files[f.stem] = f
    print(f"Found {len(mask_files)} mask files")
    
    # Prepare config
    config = {
        'use_alpha': use_alpha,
        'margin_size': margin_size,
        'canny_low': canny_low,
        'canny_high': canny_high,
        'sobel_kernel': sobel_kernel,
        'component_mode': component_mode,
        'output_folder': output_path
    }
    
    # Prepare tasks
    tasks = []
    for view_id, image_path in view_mapping.items():
        if not Path(image_path).exists():
            print(f"  Warning: Image not found: {image_path}")
            continue
        
        # Find corresponding mask
        if view_id in mask_files:
            mask_path = mask_files[view_id]
        else:
            print(f"  Warning: No mask for view {view_id}")
            continue
        
        # Check if output exists
        output_file = output_path / f"{view_id}.png"
        if not force and output_file.exists():
            continue
        
        tasks.append((view_id, image_path, str(mask_path), config))
    
    if not tasks:
        print("No images to process")
        return
    
    # Determine worker count
    workers = num_workers or max(1, multiprocessing.cpu_count() - 1)
    print(f"Processing {len(tasks)} images with {workers} workers...")
    
    # Process in parallel
    stats = {'success': 0, 'skipped': 0, 'error': 0}
    
    with multiprocessing.Pool(processes=workers) as pool:
        results = list(tqdm(
            pool.imap(process_single_image, tasks),
            total=len(tasks),
            desc="Extracting contours"
        ))
    
    # Collect results and build output mapping
    contours_mapping = {}
    for result in results:
        status = result['status']
        stats[status] = stats.get(status, 0) + 1
        if status == 'error':
            tqdm.write(f"  Error {result['view_id']}: {result.get('message', 'Unknown')}")
        elif status == 'success' and 'output_path' in result:
            contours_mapping[result['view_id']] = result['output_path']
    
    # Add existing contour files (not regenerated)
    for view_id in view_mapping.keys():
        if view_id not in contours_mapping:
            existing_file = output_path / f"{view_id}.png"
            if existing_file.exists():
                contours_mapping[view_id] = str(existing_file)
    
    # Create SfMData with contour paths
    sfmdata_output = output_path / "sfmdata_contours.json"
    _save_sfmdata_with_contours(sfm_path, contours_mapping, sfmdata_output)
    
    print(f"\nResults: {stats['success']} success, {stats['skipped']} skipped, {stats['error']} errors")
    print(f"Output saved to: {output_folder}")
    print(f"SfMData with contours saved to: {sfmdata_output}")


def _save_sfmdata_with_contours(
    sfm_path: str,
    contours_mapping: Dict[str, str],
    output_path: Path
) -> None:
    """
    Create and save an SfMData file with image paths replaced by contour paths.
    
    This creates a new SfMData JSON file where each view's 'path' field
    points to the generated contour image instead of the original image.
    Only views that have corresponding contours are included.
    
    Args:
        sfm_path: Path to the original SfMData file
        contours_mapping: Dict mapping view_id to contour image path
        output_path: Output path for the new SfMData JSON file
    """
    # Load original SfMData as dict
    sfmdata = load_sfmdata_json(sfm_path)
    
    # Create a deep copy to avoid modifying original
    sfmdata_contours = copy.deepcopy(sfmdata)
    
    # Filter views and update paths
    new_views = []
    for view in sfmdata_contours.get('views', []):
        view_id = str(view.get('viewId'))
        if view_id in contours_mapping:
            # Update path to contour image
            view['path'] = contours_mapping[view_id]
            new_views.append(view)
    
    sfmdata_contours['views'] = new_views
    
    # Filter poses to keep only those referenced by remaining views
    used_pose_ids = {view.get('poseId') for view in new_views}
    sfmdata_contours['poses'] = [
        pose for pose in sfmdata_contours.get('poses', [])
        if pose.get('poseId') in used_pose_ids
    ]
    
    # Filter intrinsics to keep only those referenced by remaining views
    used_intrinsic_ids = {view.get('intrinsicId') for view in new_views}
    sfmdata_contours['intrinsics'] = [
        intr for intr in sfmdata_contours.get('intrinsics', [])
        if intr.get('intrinsicId') in used_intrinsic_ids
    ]
    
    # Save the new SfMData
    save_sfmdata_json(sfmdata_contours, str(output_path))


def main():
    parser = argparse.ArgumentParser(
        description='Generate Canny edge contours from SfMData images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with separate mask files
  pyav-contours --sfm sfmdata.json --masks masks/ --output contours/

  # Use alpha channel from mask files
  pyav-contours --sfm sfmdata.json --masks masks/ --output contours/ --use-alpha

  # Custom Canny thresholds
  pyav-contours --sfm sfmdata.json --masks masks/ --output contours/ \\
                --canny-low 40 --canny-high 120
        """
    )
    parser.add_argument('--sfm', '-s', required=True, help='SfMData JSON file')
    parser.add_argument('--masks', '-m', required=True, help='Masks folder')
    parser.add_argument('--output', '-o', required=True, help='Output folder')
    parser.add_argument('--use-alpha', action='store_true', 
                       help='Extract mask from alpha channel')
    parser.add_argument('--margin-size', type=int, default=20,
                       help='Margin size in pixels (default: 20)')
    parser.add_argument('--canny-low', type=int, default=50,
                       help='Canny low threshold (default: 50)')
    parser.add_argument('--canny-high', type=int, default=150,
                       help='Canny high threshold (default: 150)')
    parser.add_argument('--sobel-kernel', type=int, default=3, choices=[3, 5, 7],
                       help='Sobel kernel size (default: 3)')
    parser.add_argument('--component-mode', default='hybrid',
                       choices=['center_point', 'smallest_area', 'hybrid'],
                       help='Connected component filter mode (default: hybrid)')
    parser.add_argument('--num-workers', type=int, default=None,
                       help='Number of parallel workers')
    parser.add_argument('--force', action='store_true',
                       help='Regenerate existing outputs')
    
    args = parser.parse_args()
    
    process_contours(
        sfm_path=args.sfm,
        masks_folder=args.masks,
        output_folder=args.output,
        use_alpha=args.use_alpha,
        margin_size=args.margin_size,
        canny_low=args.canny_low,
        canny_high=args.canny_high,
        sobel_kernel=args.sobel_kernel,
        component_mode=args.component_mode,
        num_workers=args.num_workers,
        force=args.force
    )


if __name__ == '__main__':
    main()
