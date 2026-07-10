"""
Integration tests for RealityCapture <-> SfMData conversion with point clouds.
"""

import json

import numpy as np
import pytest

from pyalicevisionlib.pointcloud import (
    load_point_cloud,
    save_point_cloud_ply,
    rc_points_to_av,
)
from pyalicevisionlib.scripts.rc_to_sfmdata import convert_rc_to_sfmdata
from pyalicevisionlib.scripts.sfmdata_to_rc import convert_sfmdata_to_rc


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

XMP_CONTENT = """<x:xmpmeta xmlns:x="adobe:ns:meta/">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about=""
    xmlns:xcr="http://www.capturingreality.com/ns/xcr/1.1#"
    xcr:Version="3"
    xcr:PosePrior="locked"
    xcr:Coordinates="absolute"
    xcr:DistortionModel="brown3"
    xcr:FocalLength35mm="50"
    xcr:Skew="0"
    xcr:AspectRatio="1"
    xcr:PrincipalPointU="0.001"
    xcr:PrincipalPointV="-0.002"
    xcr:CalibrationPrior="locked">
   <xcr:Rotation>1 0 0 0 1 0 0 0 1</xcr:Rotation>
   <xcr:Position>1 2 3</xcr:Position>
   <xcr:DistortionCoeficients>0 0 0 0 0 0</xcr:DistortionCoeficients>
  </rdf:Description>
 </rdf:RDF>
</x:xmpmeta>
"""


@pytest.fixture
def rc_export_folder(tmp_path):
    """Fake RealityCapture export: one XMP + one image + a point cloud."""
    import cv2

    xmp_folder = tmp_path / "xmp"
    images_folder = tmp_path / "images"
    xmp_folder.mkdir()
    images_folder.mkdir()

    (xmp_folder / "img_001.xmp").write_text(XMP_CONTENT)
    cv2.imwrite(str(images_folder / "img_001.png"), np.zeros((6, 8, 3), dtype=np.uint8))

    points_rc = np.array([[1.0, 2.0, 3.0], [-4.0, 5.0, -6.0]])
    colors = np.array([[10, 20, 30], [200, 210, 220]], dtype=np.uint8)
    cloud_path = tmp_path / "cloud.ply"
    save_point_cloud_ply(str(cloud_path), points_rc, colors)

    return {
        "xmp_folder": xmp_folder,
        "images_folder": images_folder,
        "cloud_path": cloud_path,
        "points_rc": points_rc,
        "colors": colors,
    }


@pytest.fixture
def sfmdata_with_structure(tmp_path):
    """Minimal SfMData JSON containing one posed view and two landmarks."""
    points_av = np.array([[1.0, -2.0, -3.0], [0.5, 0.25, -0.75]])
    colors = np.array([[10, 20, 30], [40, 50, 60]], dtype=np.uint8)

    sfmdata = {
        "version": ["1", "2", "13"],
        "views": [{
            "viewId": "100",
            "poseId": "100",
            "frameId": "1",
            "intrinsicId": "200",
            "path": str(tmp_path / "missing.png"),
            "width": "8",
            "height": "6",
        }],
        "intrinsics": [{
            "intrinsicId": "200",
            "width": "8",
            "height": "6",
            "sensorWidth": "36.0",
            "sensorHeight": "24.0",
            "focalLength": "50.0",
            "principalPoint": ["0.0", "0.0"],
            "distortionType": "none",
            "distortionParams": [],
        }],
        "poses": [{
            "poseId": "100",
            "pose": {
                "transform": {
                    "rotation": ["1", "0", "0", "0", "1", "0", "0", "0", "1"],
                    "center": ["1", "2", "3"],
                },
                "locked": "0",
            },
        }],
        "structure": [
            {
                "landmarkId": str(i),
                "descType": "sift",
                "color": [str(c) for c in colors[i]],
                "X": [str(x) for x in points_av[i]],
                "observations": [],
            }
            for i in range(len(points_av))
        ],
    }

    path = tmp_path / "sfmdata.json"
    path.write_text(json.dumps(sfmdata))
    return {"path": path, "points_av": points_av, "colors": colors}


# ---------------------------------------------------------------------------
# RC -> SfMData with point cloud
# ---------------------------------------------------------------------------

class TestRCToSfMDataPointCloud:
    def test_point_cloud_imported_as_structure(self, tmp_path, rc_export_folder):
        output = tmp_path / "sfmdata.json"
        convert_rc_to_sfmdata(
            xmp_folder=str(rc_export_folder["xmp_folder"]),
            images_folder=str(rc_export_folder["images_folder"]),
            output_path=str(output),
            point_cloud=str(rc_export_folder["cloud_path"]),
        )

        data = json.loads(output.read_text())
        structure = data.get("structure", [])
        assert len(structure) == 2

        points = np.array([[float(v) for v in lm["X"]] for lm in structure])
        expected = rc_points_to_av(rc_export_folder["points_rc"])
        np.testing.assert_allclose(points, expected, rtol=1e-6)

        colors = np.array([[int(v) for v in lm["color"]] for lm in structure])
        np.testing.assert_array_equal(colors, rc_export_folder["colors"])

    def test_no_point_cloud_no_structure(self, tmp_path, rc_export_folder):
        output = tmp_path / "sfmdata.json"
        convert_rc_to_sfmdata(
            xmp_folder=str(rc_export_folder["xmp_folder"]),
            images_folder=str(rc_export_folder["images_folder"]),
            output_path=str(output),
        )
        data = json.loads(output.read_text())
        assert data.get("structure", []) == []

    def test_missing_point_cloud_raises(self, tmp_path, rc_export_folder):
        with pytest.raises(FileNotFoundError):
            convert_rc_to_sfmdata(
                xmp_folder=str(rc_export_folder["xmp_folder"]),
                images_folder=str(rc_export_folder["images_folder"]),
                output_path=str(tmp_path / "sfmdata.json"),
                point_cloud=str(tmp_path / "missing.ply"),
            )


# ---------------------------------------------------------------------------
# SfMData -> RC with point cloud
# ---------------------------------------------------------------------------

class TestSfMDataToRCPointCloud:
    def test_structure_exported_as_ply(self, tmp_path, sfmdata_with_structure):
        output_folder = tmp_path / "rc_export"
        stats = convert_sfmdata_to_rc(
            sfmdata_path=str(sfmdata_with_structure["path"]),
            output_folder=str(output_folder),
            copy_images=False,
        )

        ply_path = output_folder / "point_cloud.ply"
        assert ply_path.exists()
        assert stats["points_exported"] == 2

        points, colors = load_point_cloud(str(ply_path))
        # Exported points must be back in RC coordinates (Y/Z flipped from AV)
        expected = sfmdata_with_structure["points_av"] * np.array([1.0, -1.0, -1.0])
        np.testing.assert_allclose(points, expected, rtol=1e-6)
        np.testing.assert_array_equal(colors, sfmdata_with_structure["colors"])

    def test_point_cloud_export_disabled(self, tmp_path, sfmdata_with_structure):
        output_folder = tmp_path / "rc_export"
        stats = convert_sfmdata_to_rc(
            sfmdata_path=str(sfmdata_with_structure["path"]),
            output_folder=str(output_folder),
            copy_images=False,
            export_point_cloud=False,
        )
        assert not (output_folder / "point_cloud.ply").exists()
        assert stats["points_exported"] == 0

    def test_no_structure_no_ply(self, tmp_path, sfmdata_with_structure):
        data = json.loads(sfmdata_with_structure["path"].read_text())
        data["structure"] = []
        path = tmp_path / "no_structure.json"
        path.write_text(json.dumps(data))

        output_folder = tmp_path / "rc_export"
        stats = convert_sfmdata_to_rc(
            sfmdata_path=str(path),
            output_folder=str(output_folder),
            copy_images=False,
        )
        assert not (output_folder / "point_cloud.ply").exists()
        assert stats["points_exported"] == 0
