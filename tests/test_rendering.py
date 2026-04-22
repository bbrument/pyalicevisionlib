"""
Tests for the rendering module (normal maps and masks from mesh + cameras).
"""

import numpy as np
import pytest
import trimesh

from pyalicevisionlib.camera import Camera
from pyalicevisionlib.rendering import (
    _compute_barycentric,
    _generate_rays,
    render_normal_map,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def identity_camera():
    """Camera at origin looking along +Z, 100x100, 90-degree FOV."""
    return Camera(
        view_id="test_cam",
        width=100,
        height=100,
        focal_length_mm=18.0,
        sensor_width=36.0,
        principal_point=np.array([0.0, 0.0]),
        center=np.array([0.0, 0.0, 0.0]),
        rotation_cam2world=np.eye(3),
    )


@pytest.fixture
def front_plane_mesh():
    """A flat quad (two triangles) at z=5, spanning [-2,2] x [-2,2].

    Normal of both triangles points towards -Z (towards camera at origin).
    """
    vertices = np.array([
        [-2, -2, 5],
        [ 2, -2, 5],
        [ 2,  2, 5],
        [-2,  2, 5],
    ], dtype=np.float64)
    faces = np.array([
        [0, 2, 1],
        [0, 3, 2],
    ], dtype=np.int32)
    return trimesh.Trimesh(vertices=vertices, faces=faces)


@pytest.fixture
def cube_mesh():
    """A unit cube centered at (0, 0, 3)."""
    mesh = trimesh.creation.box(extents=[2, 2, 2])
    mesh.apply_translation([0, 0, 3])
    return mesh


# ---------------------------------------------------------------------------
# _compute_barycentric
# ---------------------------------------------------------------------------

class TestComputeBarycentric:

    def test_vertex_positions(self):
        """Barycentric coords at each vertex should be (1,0,0), (0,1,0), (0,0,1)."""
        tri = np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0]]], dtype=np.float64)
        for i in range(3):
            bary = _compute_barycentric(tri, tri[:, i])
            expected = np.zeros(3)
            expected[i] = 1.0
            np.testing.assert_allclose(bary[0], expected, atol=1e-10)

    def test_centroid(self):
        """Barycentric coords at centroid should be ~(1/3, 1/3, 1/3)."""
        tri = np.array([[[0, 0, 0], [3, 0, 0], [0, 3, 0]]], dtype=np.float64)
        centroid = tri.mean(axis=1)
        bary = _compute_barycentric(tri, centroid)
        np.testing.assert_allclose(bary[0], [1/3, 1/3, 1/3], atol=1e-10)

    def test_batch(self):
        """Multiple triangles at once."""
        tris = np.array([
            [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
            [[0, 0, 0], [2, 0, 0], [0, 2, 0]],
        ], dtype=np.float64)
        points = tris.mean(axis=1)
        bary = _compute_barycentric(tris, points)
        assert bary.shape == (2, 3)
        np.testing.assert_allclose(bary, 1/3, atol=1e-10)

    def test_sum_to_one(self):
        """Barycentric coordinates must sum to 1."""
        tri = np.array([[[0, 0, 0], [4, 0, 0], [0, 4, 0]]], dtype=np.float64)
        point = np.array([[1.0, 1.0, 0.0]])
        bary = _compute_barycentric(tri, point)
        np.testing.assert_allclose(bary.sum(), 1.0, atol=1e-10)


# ---------------------------------------------------------------------------
# _generate_rays
# ---------------------------------------------------------------------------

class TestGenerateRays:

    def test_output_shape(self, identity_camera):
        """Should produce H*W rays."""
        origins, directions = _generate_rays(identity_camera)
        n = identity_camera.height * identity_camera.width
        assert origins.shape == (n, 3)
        assert directions.shape == (n, 3)

    def test_origins_at_camera_center(self, identity_camera):
        """All ray origins should equal camera center."""
        origins, _ = _generate_rays(identity_camera)
        for i in range(3):
            np.testing.assert_allclose(origins[:, i], identity_camera.center[i], atol=1e-12)

    def test_directions_normalized(self, identity_camera):
        """All ray directions should be unit vectors."""
        _, directions = _generate_rays(identity_camera)
        norms = np.linalg.norm(directions, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-10)

    def test_center_ray_along_z(self, identity_camera):
        """Center pixel ray should point along +Z for identity camera."""
        _, directions = _generate_rays(identity_camera)
        H, W = identity_camera.height, identity_camera.width
        center_idx = (H // 2) * W + (W // 2)
        center_dir = directions[center_idx]
        # Z should dominate, X and Y near zero
        assert center_dir[2] > 0.9
        assert abs(center_dir[0]) < 0.05
        assert abs(center_dir[1]) < 0.05

    def test_right_pixel_has_positive_x(self, identity_camera):
        """A pixel to the right of center should have positive X direction."""
        _, directions = _generate_rays(identity_camera)
        H, W = identity_camera.height, identity_camera.width
        right_idx = (H // 2) * W + (W - 1)
        assert directions[right_idx][0] > 0


# ---------------------------------------------------------------------------
# render_normal_map
# ---------------------------------------------------------------------------

class TestRenderNormalMap:

    def test_output_shapes(self, identity_camera, front_plane_mesh):
        """Normal map and mask should have correct shapes."""
        normal_map, mask = render_normal_map(front_plane_mesh, identity_camera)
        H, W = identity_camera.height, identity_camera.width
        assert normal_map.shape == (H, W, 3)
        assert normal_map.dtype == np.uint16
        assert mask.shape == (H, W)
        assert mask.dtype == bool

    def test_mask_has_hits(self, identity_camera, front_plane_mesh):
        """At least some pixels should hit the mesh."""
        _, mask = render_normal_map(front_plane_mesh, identity_camera)
        assert mask.any(), "No pixel hit the mesh"

    def test_mask_center_hit(self, identity_camera, front_plane_mesh):
        """Center pixel should hit the front plane."""
        _, mask = render_normal_map(front_plane_mesh, identity_camera)
        cy, cx = identity_camera.height // 2, identity_camera.width // 2
        assert mask[cy, cx], "Center pixel should hit the front plane"

    def test_normal_towards_camera(self, identity_camera, front_plane_mesh):
        """For a plane facing the camera, B channel should be high (towards camera)."""
        normal_map, mask = render_normal_map(front_plane_mesh, identity_camera)
        # Get normal at center where mask is True
        cy, cx = identity_camera.height // 2, identity_camera.width // 2
        if mask[cy, cx]:
            blue = normal_map[cy, cx, 2]
            # B = towards camera => normal_z in [-1,1] mapped to [0,65535]
            # Plane at z=5 facing camera => normal_cam_z (after flip) ≈ +1 => ~65535
            assert blue > 50000, f"Blue channel {blue} should be high for surface facing camera"

    def test_normal_encoding_range(self, identity_camera, front_plane_mesh):
        """Normal map values should stay within [0, 65535]."""
        normal_map, mask = render_normal_map(front_plane_mesh, identity_camera)
        hit_values = normal_map[mask]
        assert hit_values.min() >= 0
        assert hit_values.max() <= 65535

    def test_background_is_midpoint(self, identity_camera, front_plane_mesh):
        """Pixels without hits should encode zero vector as midpoint (gray)."""
        normal_map, mask = render_normal_map(front_plane_mesh, identity_camera)
        bg = normal_map[~mask]
        midpoint = int(0.5 * 65535)
        if len(bg) > 0:
            np.testing.assert_array_equal(bg, midpoint)

    def test_cube_mask_center(self, identity_camera, cube_mesh):
        """Center of a cube in front of camera should be hit."""
        _, mask = render_normal_map(cube_mesh, identity_camera)
        cy, cx = identity_camera.height // 2, identity_camera.width // 2
        assert mask[cy, cx]

    def test_cube_front_face_normal(self, identity_camera, cube_mesh):
        """Front face of cube should have normal pointing towards camera (high B)."""
        normal_map, mask = render_normal_map(cube_mesh, identity_camera)
        cy, cx = identity_camera.height // 2, identity_camera.width // 2
        if mask[cy, cx]:
            r, g, b = normal_map[cy, cx]
            midpoint = 65535 // 2
            # R ≈ midpoint (no left/right), G ≈ midpoint (no up/down), B >> midpoint
            assert abs(r - midpoint) < 5000, f"R={r} should be near midpoint"
            assert abs(g - midpoint) < 5000, f"G={g} should be near midpoint"
            assert b > 50000, f"B={b} should be high (towards camera)"

    def test_chunk_size_consistency(self, identity_camera, front_plane_mesh):
        """Different chunk sizes should produce identical results."""
        nm1, m1 = render_normal_map(front_plane_mesh, identity_camera, chunk_size=500, samples=1)
        nm2, m2 = render_normal_map(front_plane_mesh, identity_camera, chunk_size=100_000, samples=1)
        np.testing.assert_array_equal(m1, m2)
        np.testing.assert_array_equal(nm1, nm2)

    def test_multisampling_more_hits(self, identity_camera, front_plane_mesh):
        """Multi-sampling should produce at least as many mask hits as single sample."""
        _, mask_1 = render_normal_map(front_plane_mesh, identity_camera, samples=1)
        _, mask_3 = render_normal_map(front_plane_mesh, identity_camera, samples=3)
        # 3x3 samples covers more sub-pixel positions, so mask should be >= single
        assert mask_3.sum() >= mask_1.sum()

    def test_multisampling_same_center_normal(self, identity_camera, front_plane_mesh):
        """Center normal should be similar between single and multi-sample."""
        nm1, _ = render_normal_map(front_plane_mesh, identity_camera, samples=1)
        nm3, _ = render_normal_map(front_plane_mesh, identity_camera, samples=3)
        cy, cx = identity_camera.height // 2, identity_camera.width // 2
        # At center of a flat plane, all samples agree -> same result
        np.testing.assert_allclose(nm1[cy, cx], nm3[cy, cx], atol=1)

    def test_rotated_camera(self, front_plane_mesh):
        """Camera rotated 90 deg around Z: right becomes up, normals should adapt."""
        angle = np.pi / 2
        R = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle),  np.cos(angle), 0],
            [0,              0,             1],
        ])
        cam = Camera(
            view_id="rotated",
            width=100,
            height=100,
            focal_length_mm=18.0,
            sensor_width=36.0,
            principal_point=np.array([0.0, 0.0]),
            center=np.array([0.0, 0.0, 0.0]),
            rotation_cam2world=R,
        )
        normal_map, mask = render_normal_map(front_plane_mesh, cam)
        assert mask.any(), "Rotated camera should still hit the plane"
        # B channel at center should still be high (plane faces camera regardless of rotation)
        cy, cx = cam.height // 2, cam.width // 2
        if mask[cy, cx]:
            assert normal_map[cy, cx, 2] > 50000

    def test_no_mesh_intersection(self):
        """Camera pointing away from mesh should produce empty mask."""
        cam = Camera(
            view_id="away",
            width=50,
            height=50,
            focal_length_mm=18.0,
            sensor_width=36.0,
            principal_point=np.array([0.0, 0.0]),
            center=np.array([0.0, 0.0, 0.0]),
            rotation_cam2world=np.eye(3),
        )
        # Mesh behind the camera (z = -5)
        verts = np.array([[-1, -1, -5], [1, -1, -5], [0, 1, -5]], dtype=np.float64)
        faces = np.array([[0, 1, 2]])
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        _, mask = render_normal_map(mesh, cam)
        assert not mask.any(), "No pixel should hit mesh behind camera"
