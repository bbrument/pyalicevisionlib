"""
SfMData loading and saving utilities.

Provides loading and saving of camera data from AliceVision SfMData files.
Uses pyalicevision bindings when available for full format support (.sfm, .abc, .json),
falls back to JSON parsing only when pyalicevision is not installed.

Coordinate System Notes:
    AliceVision/Meshroom uses a coordinate system that may differ from 
    standard conventions (e.g., OpenGL Y-up). We apply a transformation 
    that flips Y and Z axes on the camera poses:
    
        world_correction = diag([1, -1, -1])
        center_corrected = world_correction @ center
        R_cam2world_corrected = world_correction @ R_cam2world
    
    This makes the cameras consistent with a Y-up world where the mesh is 
    in the "correct" orientation.
"""

import json
import numpy as np
from typing import List, Dict, Optional, Union, Any, Tuple
from pathlib import Path

from .camera import Camera

# Optional pyalicevision import
try:
    from pyalicevision import sfmData as avSfmData
    from pyalicevision import sfmDataIO as avSfmDataIO
    from pyalicevision import camera as avCamera
    from pyalicevision import geometry as avGeometry
    HAS_PYALICEVISION = True
except ImportError:
    HAS_PYALICEVISION = False

# World coordinate correction matrix (flip Y and Z) - 3x3 version
WORLD_CORRECTION = np.diag([1.0, -1.0, -1.0])

# World coordinate correction matrix - 4x4 homogeneous version
WORLD_CORRECTION_4x4 = np.diag([1.0, -1.0, -1.0, 1.0])


# =============================================================================
# SfMDataWrapper - Unified interface for SfMData handling
# =============================================================================

class SfMDataWrapper:
    """
    Unified wrapper for SfMData handling.
    
    Uses pyalicevision native objects when available, falls back to dict-based
    JSON parsing otherwise. Provides a consistent API regardless of backend.
    
    Examples:
        # Load SfMData
        sfm = SfMDataWrapper.load("sfmdata.json")
        
        # Access data
        cameras = sfm.get_cameras()
        views = sfm.get_views()
        
        # Modify and save
        sfm.save("output.abc")  # Uses pyalicevision if available
        sfm.save("output.json")  # Always works
    """
    
    def __init__(self, native_sfm: Optional[Any] = None, json_data: Optional[Dict] = None):
        """
        Initialize wrapper with either native pyalicevision SfMData or JSON dict.
        
        Args:
            native_sfm: pyalicevision.sfmData.SfMData object (preferred)
            json_data: Dictionary from JSON parsing (fallback)
        """
        self._native = native_sfm
        self._json_data = json_data
        
        if native_sfm is None and json_data is None:
            raise ValueError("Either native_sfm or json_data must be provided")
    
    @property
    def has_native(self) -> bool:
        """Check if using native pyalicevision backend."""
        return self._native is not None
    
    @property
    def native(self) -> Any:
        """
        Get the native pyalicevision SfMData object.
        
        Returns:
            pyalicevision.sfmData.SfMData object
            
        Raises:
            RuntimeError: If not using native backend
        """
        if self._native is None:
            raise RuntimeError(
                "Native SfMData not available. pyalicevision is required for this operation."
            )
        return self._native
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'SfMDataWrapper':
        """
        Create a wrapper from a dictionary (JSON-like structure).
        
        This is useful when creating SfMData programmatically (e.g., from XMP files).
        When save() is called, pyalicevision will normalize the structure.
        
        Args:
            data: Dictionary with SfMData structure (views, poses, intrinsics, etc.)
            
        Returns:
            SfMDataWrapper instance
        """
        return cls(json_data=data)
    
    @classmethod
    def load(cls, path: str) -> 'SfMDataWrapper':
        """
        Load SfMData from file.
        
        Uses pyalicevision for all formats when available (.json, .sfm, .abc).
        Falls back to JSON parsing only when pyalicevision is not installed.
        
        Args:
            path: Path to SfMData file
            
        Returns:
            SfMDataWrapper instance
            
        Raises:
            ValueError: If file cannot be loaded
            ImportError: If binary format requires pyalicevision but not available
        """
        path_obj = Path(path)
        suffix = path_obj.suffix.lower()
        
        # Binary formats always require pyalicevision
        if suffix in ['.sfm', '.abc'] and not HAS_PYALICEVISION:
            raise ImportError(
                f"pyalicevision is required to load {suffix} files. "
                "See docs/pyalicevision_setup.md for installation instructions."
            )
        
        # Use pyalicevision when available (for ALL formats including .json)
        if HAS_PYALICEVISION:
            sfm = avSfmData.SfMData()
            if not avSfmDataIO.load(sfm, str(path), avSfmDataIO.ALL):
                raise ValueError(f"Failed to load SfMData: {path}")
            return cls(native_sfm=sfm)
        
        # Fallback: JSON parsing only
        if suffix != '.json':
            raise ImportError(
                f"pyalicevision is required to load {suffix} files. "
                "Only .json format is supported without pyalicevision."
            )
        
        with open(path, 'r') as f:
            json_data = json.load(f)
        return cls(json_data=json_data)
    
    def save(self, path: str) -> None:
        """
        Save SfMData to file.
        
        Uses pyalicevision when available for all formats.
        Falls back to JSON when pyalicevision is not available (forces .json extension).
        
        When saving JSON data with pyalicevision available, the data is first
        saved to a temp JSON, loaded with pyalicevision (which normalizes the
        structure and adds missing fields), then saved to the final path.
        
        Args:
            path: Output path. If pyalicevision not available and path is .abc/.sfm,
                  will automatically switch to .json with a warning.
        """
        path_obj = Path(path)
        suffix = path_obj.suffix.lower()
        
        # Use pyalicevision when available with native data
        if self._native is not None:
            if not avSfmDataIO.save(self._native, str(path), avSfmDataIO.ALL):
                raise ValueError(f"Failed to save SfMData: {path}")
            return
        
        # If we have JSON data but pyalicevision is available, convert through pyav
        # This normalizes the structure and adds missing fields like undistortionType
        if self._json_data is not None and HAS_PYALICEVISION:
            import tempfile
            import os
            
            # Save JSON to temp file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(self._json_data, f, indent=4)
                temp_json = f.name
            
            try:
                # Load with pyalicevision (normalizes structure)
                sfm = avSfmData.SfMData()
                if avSfmDataIO.load(sfm, temp_json, avSfmDataIO.ALL):
                    # Save with pyalicevision to final path
                    if not avSfmDataIO.save(sfm, str(path), avSfmDataIO.ALL):
                        raise ValueError(f"Failed to save SfMData: {path}")
                    # Update internal native reference
                    self._native = sfm
                    return
                else:
                    # pyalicevision failed to load, fall back to direct JSON save
                    import warnings
                    warnings.warn(
                        f"pyalicevision failed to parse JSON, saving directly: {path}"
                    )
            finally:
                os.unlink(temp_json)
        
        # Fallback: save as JSON (force .json extension if needed)
        if suffix in ['.sfm', '.abc']:
            new_path = path_obj.with_suffix('.json')
            import warnings
            warnings.warn(
                f"pyalicevision not available, saving as JSON instead: {new_path}"
            )
            path = str(new_path)
        
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self._json_data, f, indent=4)
    
    def get_cameras(self, apply_world_correction: bool = True) -> List[Camera]:
        """
        Extract Camera objects from SfMData.
        
        Args:
            apply_world_correction: If True, apply Y/Z flip correction
            
        Returns:
            List of Camera objects with valid poses
        """
        if self._native is not None:
            cameras = self._get_cameras_from_native()
        else:
            cameras = self._get_cameras_from_json()
        
        if apply_world_correction:
            cameras = _apply_world_correction(cameras)
        
        return cameras
    
    def _get_cameras_from_native(self) -> List[Camera]:
        """
        Extract cameras from native pyalicevision SfMData.
        
        Note: Due to buffer reuse issues in pyalicevision Python bindings,
        we extract cameras by first converting to JSON dict, then parsing.
        This is more reliable than extracting values directly from the C++ objects.
        """
        # Convert to JSON dict and use JSON parser (avoids pyalicevision buffer issues)
        json_data = self.as_dict()
        # Temporarily set json_data to extract cameras
        original_json = self._json_data
        self._json_data = json_data
        try:
            cameras = self._get_cameras_from_json()
        finally:
            self._json_data = original_json
        return cameras
    
    def _get_cameras_from_json(self) -> List[Camera]:
        """Extract cameras from JSON dict."""
        # Parse intrinsics into lookup dict
        intrinsics_dict = {}
        for intr in self._json_data.get('intrinsics', []):
            intrinsics_dict[intr['intrinsicId']] = intr
        
        # Parse poses into lookup dict
        poses_dict = {}
        for pose in self._json_data.get('poses', []):
            poses_dict[pose['poseId']] = pose['pose']['transform']
        
        cameras = []
        for view in self._json_data.get('views', []):
            view_id = view['viewId']
            intr_id = view['intrinsicId']
            pose_id = view['poseId']
            
            if intr_id not in intrinsics_dict or pose_id not in poses_dict:
                continue
            
            intr = intrinsics_dict[intr_id]
            transform = poses_dict[pose_id]
            
            width = int(intr['width'])
            height = int(intr['height'])
            focal_mm = float(intr['focalLength'])
            sensor_width = float(intr.get('sensorWidth', 36.0))
            
            pp = intr.get('principalPoint', ['0', '0'])
            pp_offset = np.array([float(pp[0]), float(pp[1])])
            
            rotation_flat = [float(r) for r in transform['rotation']]
            rotation_cam2world = np.array(rotation_flat).reshape(3, 3)
            
            center = np.array([float(c) for c in transform['center']])
            
            cameras.append(Camera(
                view_id=view_id,
                width=width,
                height=height,
                focal_length_mm=focal_mm,
                sensor_width=sensor_width,
                principal_point=pp_offset,
                center=center,
                rotation_cam2world=rotation_cam2world,
                image_path=view.get('path')
            ))
        
        return cameras
    
    def get_views(self) -> Union[Dict, Any]:
        """
        Get views from SfMData.
        
        Returns:
            Native views dict-like object if using pyalicevision,
            otherwise list of view dicts from JSON.
        """
        if self._native is not None:
            return self._native.getViews()
        return self._json_data.get('views', [])
    
    def get_poses(self) -> Union[Dict, Any]:
        """
        Get poses from SfMData.
        
        Returns:
            Native poses dict-like object if using pyalicevision,
            otherwise list of pose dicts from JSON.
        """
        if self._native is not None:
            return self._native.getPoses()
        return self._json_data.get('poses', [])
    
    def get_intrinsics(self) -> Union[Dict, Any]:
        """
        Get intrinsics from SfMData.
        
        Returns:
            Native intrinsics dict-like object if using pyalicevision,
            otherwise list of intrinsic dicts from JSON.
        """
        if self._native is not None:
            return self._native.getIntrinsics()
        return self._json_data.get('intrinsics', [])
    
    def get_view(self, view_id: int) -> Optional[Any]:
        """
        Get a specific view by ID.
        
        Args:
            view_id: View ID
            
        Returns:
            View object (native or dict) or None if not found
        """
        if self._native is not None:
            views = self._native.getViews()
            if view_id in views:
                return views[view_id]
            return None
        
        for view in self._json_data.get('views', []):
            if str(view.get('viewId')) == str(view_id):
                return view
        return None
    
    def is_pose_and_intrinsic_defined(self, view_id: int) -> bool:
        """Check if a view has valid pose and intrinsics."""
        if self._native is not None:
            return self._native.isPoseAndIntrinsicDefined(view_id)
        
        # JSON fallback
        view = self.get_view(view_id)
        if view is None:
            return False
        
        pose_id = view.get('poseId')
        intr_id = view.get('intrinsicId')
        
        has_pose = any(p.get('poseId') == pose_id for p in self._json_data.get('poses', []))
        has_intr = any(i.get('intrinsicId') == intr_id for i in self._json_data.get('intrinsics', []))
        
        return has_pose and has_intr
    
    def get_viewid_by_image_name(self) -> Dict[str, str]:
        """
        Build a mapping from image filename (stem) to viewId.
        
        Returns:
            Dict mapping image stem to viewId
        """
        mapping = {}
        
        if self._native is not None:
            views = self._native.getViews()
            for view_id in views:
                view = views[view_id]
                image_path = view.getImage().getImagePath()
                image_name = Path(image_path).stem
                if image_name:
                    mapping[image_name] = str(view_id)
        else:
            for view in self._json_data.get('views', []):
                image_path = view.get('path', '')
                image_name = Path(image_path).stem
                view_id = view.get('viewId')
                if image_name and view_id:
                    mapping[image_name] = str(view_id)
        
        return mapping
    
    def get_viewid_to_image_path(self) -> Dict[str, str]:
        """
        Build a mapping from viewId to image path.
        
        Returns:
            Dict mapping viewId to image path
        """
        mapping = {}
        
        if self._native is not None:
            views = self._native.getViews()
            for view_id in views:
                view = views[view_id]
                image_path = view.getImage().getImagePath()
                if image_path:
                    mapping[str(view_id)] = image_path
        else:
            for view in self._json_data.get('views', []):
                view_id = view.get('viewId')
                image_path = view.get('path')
                if view_id and image_path:
                    mapping[str(view_id)] = image_path
        
        return mapping
    
    def get_views_per_pose_id(self) -> Dict[int, List[Tuple[int, str]]]:
        """
        Group views by poseId.
        
        Returns:
            Dict mapping poseId to list of (viewId, imagePath) tuples
        """
        views_per_pose = {}
        
        if self._native is not None:
            views = self._native.getViews()
            for view_id in views:
                view = views[view_id]
                pose_id = view.getPoseId()
                image_path = view.getImage().getImagePath()
                
                if pose_id not in views_per_pose:
                    views_per_pose[pose_id] = []
                views_per_pose[pose_id].append((view_id, image_path))
        else:
            for view in self._json_data.get('views', []):
                view_id = int(view.get('viewId', 0))
                pose_id = int(view.get('poseId', view_id))
                image_path = view.get('path', '')
                
                if pose_id not in views_per_pose:
                    views_per_pose[pose_id] = []
                views_per_pose[pose_id].append((view_id, image_path))
        
        return views_per_pose
    
    def apply_transform(self, T: np.ndarray) -> 'SfMDataWrapper':
        """
        Apply a 4x4 transformation matrix to all camera poses.
        
        This method transforms camera poses so that projections remain invariant
        when the mesh is also transformed by T.
        
        The transformation T is assumed to be defined in a "normal" coordinate
        system (Y up). Meshroom uses a flipped coordinate system where Y and Z
        are inverted (WORLD_CORRECTION = diag([1, -1, -1])).
        
        The approach is:
        1. Convert Meshroom data to "normal" coordinate system (multiply by S)
        2. Apply transformation T
        3. Convert back to Meshroom coordinate system (multiply by S again)
        
        For the CENTER (a 3D point):
            C_user = S @ C_raw
            C_new_user = T @ C_user
            C_new_raw = S @ C_new_user
        
        For the ROTATION (orientation only, no scale):
            R_user = S @ R_raw
            R_new_user = R_T_pure @ R_user  (R_T_pure = T[:3,:3] / scale)
            R_new_raw = S @ R_new_user
        
        The scale is removed from the rotation because:
        - A rotation matrix must be orthonormal (det=1, columns have norm 1)
        - In projection, scale cancels out: x = K @ (s*v) / (s*z) = K @ v / z
        - A camera rotated 45° remains rotated 45° even if the world scales 10x
        
        Args:
            T: 4x4 transformation matrix (can include scale)
            
        Returns:
            New SfMDataWrapper with transformed poses (deep copy)
        """
        import copy
        from scipy.linalg import polar
        
        # Get data as dict and deep copy
        data = copy.deepcopy(self.as_dict())
        
        # Decompose T
        T_block = T[:3, :3]  # 3x3 part (includes scale * rotation)
        T_trans = T[:3, 3]   # Translation
        
        # Extract scale (assuming uniform scale)
        scale = np.linalg.norm(T_block[:, 0])
        
        # Extract pure rotation (without scale)
        if scale > 1e-9:
            T_rot_pure = T_block / scale
        else:
            T_rot_pure = T_block  # Degenerate case
        
        S = WORLD_CORRECTION
        
        # Apply transformation to each pose
        for pose_entry in data.get('poses', []):
            pose = pose_entry.get('pose', {})
            transform = pose.get('transform', {})
            
            if 'rotation' not in transform or 'center' not in transform:
                continue
            
            # Parse original pose (raw = Meshroom format)
            rot_flat = [float(x) for x in transform['rotation']]
            R_raw = np.array(rot_flat).reshape(3, 3)  # R_cam2world
            C_raw = np.array([float(x) for x in transform['center']])
            
            # === TRANSFORM CENTER ===
            # Convert to user space, apply full transform (with scale), convert back
            C_user = S @ C_raw
            C_new_user = T_block @ C_user + T_trans
            C_new_raw = S @ C_new_user
            
            # === TRANSFORM ROTATION ===
            # Convert to user space, apply pure rotation (no scale), convert back
            R_user = S @ R_raw
            R_new_user = T_rot_pure @ R_user
            R_new_raw = S @ R_new_user
            
            # Orthogonalize to ensure valid rotation matrix (handles numerical errors)
            R_new_raw, _ = polar(R_new_raw)
            
            # Ensure det = +1 (proper rotation, not reflection)
            if np.linalg.det(R_new_raw) < 0:
                R_new_raw = -R_new_raw
            
            # Update values (convert to string for JSON compatibility with Meshroom)
            transform['rotation'] = [str(x) for x in R_new_raw.flatten().tolist()]
            transform['center'] = [str(x) for x in C_new_raw.tolist()]
        
        return SfMDataWrapper.from_dict(data)
    
    def as_dict(self) -> Dict:
        """
        Get SfMData as a dictionary (JSON-compatible).
        
        Note: If using native backend, this requires saving to a temp file
        and re-reading as JSON, which may be slow for large datasets.
        
        Returns:
            Dictionary representation of SfMData
        """
        if self._json_data is not None:
            return self._json_data
        
        # Native backend: save to temp JSON and reload
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            avSfmDataIO.save(self._native, temp_path, avSfmDataIO.ALL)
            with open(temp_path, 'r') as f:
                return json.load(f)
        finally:
            os.unlink(temp_path)


# =============================================================================
# Internal helper functions
# =============================================================================

def _apply_world_correction(cameras: List[Camera]) -> List[Camera]:
    """
    Apply world coordinate correction to cameras (flip Y and Z).
    
    This transforms camera poses so that:
    - center_corrected = WORLD_CORRECTION @ center
    - R_cam2world_corrected = WORLD_CORRECTION @ R_cam2world
    """
    corrected_cameras = []
    for cam in cameras:
        center_corrected = (WORLD_CORRECTION @ cam.center).copy()
        rotation_cam2world_corrected = (WORLD_CORRECTION @ cam.rotation_cam2world).copy()
        
        corrected_cameras.append(Camera(
            view_id=cam.view_id,
            width=cam.width,
            height=cam.height,
            focal_length_mm=cam.focal_length_mm,
            sensor_width=cam.sensor_width,
            principal_point=cam.principal_point.copy(),
            center=center_corrected,
            rotation_cam2world=rotation_cam2world_corrected,
            image_path=cam.image_path
        ))
    
    return corrected_cameras


# =============================================================================
# Public convenience functions (legacy API + new unified API)
# =============================================================================

def load_sfmdata(path: str) -> SfMDataWrapper:
    """
    Load SfMData from file.
    
    This is the recommended way to load SfMData files. Returns a wrapper
    that provides a unified API regardless of pyalicevision availability.
    
    Args:
        path: Path to SfMData file (.json, .sfm, or .abc)
        
    Returns:
        SfMDataWrapper instance
        
    Example:
        sfm = load_sfmdata("cameras.json")
        cameras = sfm.get_cameras()
        sfm.save("output.abc")  # Uses pyalicevision if available
    """
    return SfMDataWrapper.load(path)


def save_sfmdata(sfm: SfMDataWrapper, path: str) -> None:
    """
    Save SfMData to file.
    
    Uses pyalicevision when available for full format support.
    Falls back to JSON when necessary.
    
    Args:
        sfm: SfMDataWrapper instance
        path: Output path
    """
    sfm.save(path)


def load_cameras_from_sfmdata(sfm_path: str) -> List[Camera]:
    """
    Load cameras from AliceVision SfMData file.
    
    Uses pyalicevision for ALL formats when available (including .json).
    Falls back to JSON parsing only when pyalicevision is not installed.
    
    Note: This function applies world coordinate correction (Y/Z flip)
    to make cameras consistent with visualization conventions.
    
    Args:
        sfm_path: Path to SfMData file (.sfm, .json, or .abc)
        
    Returns:
        List of Camera objects with valid poses
        
    Raises:
        ValueError: If file cannot be loaded
        ImportError: If binary format requires pyalicevision but not available
    """
    sfm = SfMDataWrapper.load(sfm_path)
    return sfm.get_cameras(apply_world_correction=True)


def get_camera_centers(cameras: List[Camera]) -> np.ndarray:
    """
    Extract camera centers as (N, 3) array.
    
    Args:
        cameras: List of Camera objects
        
    Returns:
        (N, 3) array of camera center positions in world coordinates
    """
    return np.array([cam.center for cam in cameras])


# =============================================================================
# Legacy JSON-only functions (kept for backward compatibility)
# =============================================================================

def load_sfmdata_json(path: str) -> Dict:
    """
    Load SfMData JSON file as a dictionary.
    
    DEPRECATED: Use load_sfmdata() instead, which handles all formats
    and uses pyalicevision when available.
    
    Args:
        path: Path to SfMData JSON file
        
    Returns:
        Dictionary with SfMData content
    """
    with open(path, 'r') as f:
        return json.load(f)


def save_sfmdata_json(data: Dict, path: str) -> None:
    """
    Save SfMData dictionary to JSON file.
    
    DEPRECATED: Use SfMDataWrapper.save() instead, which uses
    pyalicevision when available for full format support.
    
    Args:
        data: SfMData dictionary
        path: Output path
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)


def get_viewid_by_image_name(sfmdata: Dict) -> Dict[str, str]:
    """
    Build a mapping from image filename (stem) to viewId.
    
    DEPRECATED: Use SfMDataWrapper.get_viewid_by_image_name() instead.
    
    Args:
        sfmdata: SfMData dictionary
        
    Returns:
        Dict mapping image stem to viewId
    """
    mapping = {}
    for view in sfmdata.get('views', []):
        image_path = view.get('path', '')
        image_name = Path(image_path).stem
        view_id = view.get('viewId')
        if image_name and view_id:
            mapping[image_name] = view_id
    return mapping


def get_viewid_to_image_path(sfmdata: Dict) -> Dict[str, str]:
    """
    Build a mapping from viewId to image path.
    
    DEPRECATED: Use SfMDataWrapper.get_viewid_to_image_path() instead.
    
    Args:
        sfmdata: SfMData dictionary
        
    Returns:
        Dict mapping viewId to image path
    """
    mapping = {}
    for view in sfmdata.get('views', []):
        view_id = view.get('viewId')
        image_path = view.get('path')
        if view_id and image_path:
            mapping[str(view_id)] = image_path
    return mapping
