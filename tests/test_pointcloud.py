"""
Tests for the pointcloud module (PLY/XYZ I/O, RC <-> AV conversion, landmarks).
"""

import numpy as np
import pytest

from pyalicevisionlib.pointcloud import (
    load_point_cloud,
    save_point_cloud_ply,
    rc_points_to_av,
    av_points_to_rc,
    landmarks_from_points,
    points_from_landmarks,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_points():
    """Small deterministic point cloud."""
    return np.array([
        [0.0, 0.0, 0.0],
        [1.0, 2.0, 3.0],
        [-1.5, 0.5, -2.25],
        [10.0, -10.0, 5.0],
    ])


@pytest.fixture
def sample_colors():
    """Matching uint8 colors."""
    return np.array([
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
        [128, 128, 128],
    ], dtype=np.uint8)


# ---------------------------------------------------------------------------
# PLY round-trip
# ---------------------------------------------------------------------------

class TestPlyIO:
    def test_binary_roundtrip_with_colors(self, tmp_path, sample_points, sample_colors):
        path = tmp_path / "cloud.ply"
        save_point_cloud_ply(str(path), sample_points, sample_colors, binary=True)
        points, colors = load_point_cloud(str(path))
        np.testing.assert_allclose(points, sample_points, rtol=1e-6)
        np.testing.assert_array_equal(colors, sample_colors)

    def test_ascii_roundtrip_with_colors(self, tmp_path, sample_points, sample_colors):
        path = tmp_path / "cloud.ply"
        save_point_cloud_ply(str(path), sample_points, sample_colors, binary=False)
        points, colors = load_point_cloud(str(path))
        np.testing.assert_allclose(points, sample_points, rtol=1e-6)
        np.testing.assert_array_equal(colors, sample_colors)

    def test_roundtrip_without_colors(self, tmp_path, sample_points):
        path = tmp_path / "cloud.ply"
        save_point_cloud_ply(str(path), sample_points, binary=True)
        points, colors = load_point_cloud(str(path))
        np.testing.assert_allclose(points, sample_points, rtol=1e-6)
        assert colors is None

    def test_ascii_ply_with_normals_and_colors(self, tmp_path):
        """RC exports may interleave normals; loader must still find xyz + rgb."""
        content = (
            "ply\n"
            "format ascii 1.0\n"
            "element vertex 2\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "property float nx\n"
            "property float ny\n"
            "property float nz\n"
            "property uchar red\n"
            "property uchar green\n"
            "property uchar blue\n"
            "end_header\n"
            "1.0 2.0 3.0 0.0 0.0 1.0 10 20 30\n"
            "4.0 5.0 6.0 0.0 1.0 0.0 40 50 60\n"
        )
        path = tmp_path / "cloud.ply"
        path.write_text(content)
        points, colors = load_point_cloud(str(path))
        np.testing.assert_allclose(points, [[1, 2, 3], [4, 5, 6]])
        np.testing.assert_array_equal(colors, [[10, 20, 30], [40, 50, 60]])

    def test_binary_big_endian(self, tmp_path, sample_points):
        header = (
            "ply\n"
            "format binary_big_endian 1.0\n"
            f"element vertex {len(sample_points)}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "end_header\n"
        )
        path = tmp_path / "cloud.ply"
        with open(path, "wb") as f:
            f.write(header.encode("ascii"))
            f.write(sample_points.astype(">f4").tobytes())
        points, colors = load_point_cloud(str(path))
        np.testing.assert_allclose(points, sample_points, rtol=1e-6)
        assert colors is None

    def test_empty_points_raises(self, tmp_path):
        with pytest.raises(ValueError):
            save_point_cloud_ply(str(tmp_path / "empty.ply"), np.zeros((0, 3)))

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_point_cloud("/nonexistent/cloud.ply")


# ---------------------------------------------------------------------------
# XYZ / text formats
# ---------------------------------------------------------------------------

class TestXyzIO:
    def test_xyz_points_only(self, tmp_path, sample_points):
        path = tmp_path / "cloud.xyz"
        np.savetxt(path, sample_points, fmt="%.6f")
        points, colors = load_point_cloud(str(path))
        np.testing.assert_allclose(points, sample_points, rtol=1e-6)
        assert colors is None

    def test_xyz_with_rgb(self, tmp_path, sample_points, sample_colors):
        path = tmp_path / "cloud.xyz"
        data = np.hstack([sample_points, sample_colors.astype(float)])
        np.savetxt(path, data, fmt="%.6f")
        points, colors = load_point_cloud(str(path))
        np.testing.assert_allclose(points, sample_points, rtol=1e-6)
        np.testing.assert_array_equal(colors, sample_colors)

    def test_unsupported_extension_raises(self, tmp_path):
        path = tmp_path / "cloud.laz"
        path.write_bytes(b"whatever")
        with pytest.raises(ValueError):
            load_point_cloud(str(path))


# ---------------------------------------------------------------------------
# Coordinate conversion (RC <-> AV world correction)
# ---------------------------------------------------------------------------

class TestCoordinateConversion:
    def test_rc_to_av_flips_y_and_z(self):
        points = np.array([[1.0, 2.0, 3.0]])
        converted = rc_points_to_av(points)
        np.testing.assert_allclose(converted, [[1.0, -2.0, -3.0]])

    def test_conversion_is_involutive(self, sample_points):
        roundtrip = av_points_to_rc(rc_points_to_av(sample_points))
        np.testing.assert_allclose(roundtrip, sample_points)

    def test_input_not_mutated(self, sample_points):
        original = sample_points.copy()
        rc_points_to_av(sample_points)
        np.testing.assert_array_equal(sample_points, original)


# ---------------------------------------------------------------------------
# Landmarks (SfMData 'structure') conversion
# ---------------------------------------------------------------------------

class TestLandmarks:
    def test_landmarks_from_points(self, sample_points, sample_colors):
        structure = landmarks_from_points(sample_points, sample_colors)
        assert len(structure) == len(sample_points)
        first = structure[0]
        assert first["landmarkId"] == "0"
        assert first["descType"] == "sift"
        assert first["observations"] == []
        assert [float(v) for v in first["X"]] == pytest.approx(sample_points[0])
        assert [int(v) for v in first["color"]] == list(sample_colors[0])

    def test_landmarks_default_color(self, sample_points):
        structure = landmarks_from_points(sample_points)
        assert all(lm["color"] == ["255", "255", "255"] for lm in structure)

    def test_points_from_landmarks_roundtrip(self, sample_points, sample_colors):
        structure = landmarks_from_points(sample_points, sample_colors)
        points, colors = points_from_landmarks(structure)
        np.testing.assert_allclose(points, sample_points)
        np.testing.assert_array_equal(colors, sample_colors)

    def test_points_from_empty_structure(self):
        points, colors = points_from_landmarks([])
        assert points.shape == (0, 3)
        assert colors.shape == (0, 3)
