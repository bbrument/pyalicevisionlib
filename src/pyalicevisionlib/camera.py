"""
Camera class for handling intrinsics and extrinsics.

Conventions:
- AliceVision stores rotation as cam2world (camera axes in world space)
- For projection: world2cam = cam2world.T
- Principal point is stored as offset from image center in pixels
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class Camera:
    """
    Camera with intrinsics and extrinsics.
    
    This class follows AliceVision conventions:
    - rotation_cam2world: 3x3 matrix with camera axes as columns (X, Y, Z)
    - center: camera position in world coordinates
    - principal_point: offset from image center in pixels [cx_offset, cy_offset]
    
    Attributes:
        view_id: Unique view identifier
        width, height: Image dimensions in pixels
        focal_length_mm: Focal length in millimeters
        sensor_width: Sensor width in millimeters  
        principal_point: Offset from image center in pixels [cx_offset, cy_offset]
        center: Camera position in world coordinates [x, y, z]
        rotation_cam2world: 3x3 rotation matrix (camera to world)
        image_path: Optional path to source image
    
    Example:
        >>> cam = Camera(
        ...     view_id="001",
        ...     width=4000, height=3000,
        ...     focal_length_mm=50.0, sensor_width=36.0,
        ...     principal_point=np.array([0.0, 0.0]),
        ...     center=np.array([0.0, 0.0, 5.0]),
        ...     rotation_cam2world=np.eye(3)
        ... )
        >>> K = cam.get_K()
        >>> uv = cam.project_points(np.array([[0, 0, 0]]))
    """
    view_id: str
    width: int
    height: int
    focal_length_mm: float
    sensor_width: float
    principal_point: np.ndarray  # offset from center [cx_offset, cy_offset]
    center: np.ndarray  # camera position in world [x, y, z]
    rotation_cam2world: np.ndarray  # 3x3 matrix, columns = camera axes in world
    image_path: Optional[str] = None
    
    @property
    def focal_length_pixels(self) -> float:
        """Focal length in pixels: f_px = f_mm * width / sensor_width."""
        return self.focal_length_mm * self.width / self.sensor_width
    
    @property
    def cx(self) -> float:
        """Principal point X (absolute pixel coordinate from top-left)."""
        return self.width / 2.0 + self.principal_point[0]
    
    @property
    def cy(self) -> float:
        """Principal point Y (absolute pixel coordinate from top-left)."""
        return self.height / 2.0 + self.principal_point[1]
    
    @property
    def rotation_world2cam(self) -> np.ndarray:
        """World to camera rotation matrix (transpose of cam2world)."""
        return self.rotation_cam2world.T
    
    def get_K(self) -> np.ndarray:
        """
        Get 3x3 intrinsic matrix K.
        
        K = [[fx,  0, cx],
             [ 0, fy, cy],
             [ 0,  0,  1]]
        
        Returns:
            3x3 intrinsic matrix
        """
        fx = fy = self.focal_length_pixels
        return np.array([
            [fx, 0, self.cx],
            [0, fy, self.cy],
            [0, 0, 1]
        ], dtype=np.float64)
    
    def get_projection_matrix(self) -> np.ndarray:
        """
        Get 3x4 projection matrix P = K @ [R | t].
        
        Projects homogeneous world points to image coordinates:
            uv_h = P @ [X, Y, Z, 1].T
            uv = uv_h[:2] / uv_h[2]
        
        Returns:
            3x4 projection matrix
        """
        R_w2c = self.rotation_world2cam
        t_w2c = -R_w2c @ self.center
        Rt = np.hstack([R_w2c, t_w2c.reshape(3, 1)])
        return self.get_K() @ Rt
    
    def project_points(self, points_3d: np.ndarray) -> np.ndarray:
        """
        Project 3D world points to 2D image coordinates.
        
        Args:
            points_3d: (N, 3) array of 3D points in world coordinates
            
        Returns:
            (N, 2) array of 2D points in pixel coordinates [u, v]
        """
        # Transform to camera coordinates: P_cam = R_w2c @ (P_world - C)
        points_centered = points_3d - self.center
        points_cam = (self.rotation_world2cam @ points_centered.T).T
        
        # Perspective projection
        Z = points_cam[:, 2]
        Z = np.where(np.abs(Z) < 1e-6, 1e-6, Z)  # Avoid division by zero
        
        fx = fy = self.focal_length_pixels
        u = fx * points_cam[:, 0] / Z + self.cx
        v = fy * points_cam[:, 1] / Z + self.cy
        
        return np.column_stack([u, v])
    
    def get_look_direction(self) -> np.ndarray:
        """
        Get camera look direction (Z axis) in world coordinates.
        
        This is the third column of the cam2world rotation matrix.
        
        Returns:
            (3,) unit vector pointing in camera's viewing direction
        """
        return self.rotation_cam2world[:, 2]
    
    def get_camera_axes(self) -> tuple:
        """
        Get camera coordinate axes in world coordinates.
        
        Returns:
            Tuple of (right, up, forward) unit vectors in world coordinates
        """
        return (
            self.rotation_cam2world[:, 0],  # X - right
            self.rotation_cam2world[:, 1],  # Y - up  
            self.rotation_cam2world[:, 2],  # Z - forward (look direction)
        )
    
    def world_to_camera(self, points_3d: np.ndarray) -> np.ndarray:
        """
        Transform points from world to camera coordinates.
        
        Args:
            points_3d: (N, 3) array of 3D points in world coordinates
            
        Returns:
            (N, 3) array of 3D points in camera coordinates
        """
        points_centered = points_3d - self.center
        return (self.rotation_world2cam @ points_centered.T).T
    
    def camera_to_world(self, points_cam: np.ndarray) -> np.ndarray:
        """
        Transform points from camera to world coordinates.
        
        Args:
            points_cam: (N, 3) array of 3D points in camera coordinates
            
        Returns:
            (N, 3) array of 3D points in world coordinates
        """
        return (self.rotation_cam2world @ points_cam.T).T + self.center
