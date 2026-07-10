# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Point cloud module** (`pointcloud.py`): dependency-free PLY (ascii/binary) and XYZ/TXT/CSV/PTS
  point cloud loading, PLY writing, RC ↔ AliceVision coordinate conversion, and conversion between
  point arrays and SfMData landmarks (`structure` section)
- **RC → SfMData** (`pyav-rc2sfm`, `RCToSfMData` node): new `--point-cloud` option to import a point
  cloud exported from RealityCapture (with the "Same as XMP" coordinate system) as SfMData landmarks
- **SfMData → RC** (`pyav-sfm2rc`, `SfMDataToRC` node): SfMData landmarks are now exported as
  `point_cloud.ply` next to the XMP files (disable with `--no-point-cloud`)

## [0.1.0] - 2024-01-22

### Added
- Initial release
- **Camera module**: `Camera` class with intrinsics/extrinsics, projection, and coordinate transforms
- **SfMData module**: Unified `SfMDataWrapper` supporting JSON and pyalicevision backends
- **Mesh module**: Mesh loading with trimesh, point sampling
- **Image module**: Unified image I/O with OIIO/OpenCV/PIL fallbacks, EXR/HDR support
- **Utils module**: Point cloud operations, mesh vertex filtering, transformation utilities
- **Evaluation module**: Chamfer distance, precision/recall, visibility-based mesh cleanup
- **Visualization module**: 3D camera and mesh plotting with matplotlib

### CLI Tools
- `pyav-visualize`: Visualize cameras and meshes from SfMData
- `pyav-evaluate`: Evaluate mesh reconstruction quality
- `pyav-rc2sfm`: Convert RealityCapture XMP to AliceVision SfMData
- `pyav-transform`: Apply 4x4 transformations to meshes and cameras
- `pyav-contours`: Extract Canny edge contours from images

### Documentation
- README with installation, quick start, and API reference
- pyalicevision setup guide in docs/
