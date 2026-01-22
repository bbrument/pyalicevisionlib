# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
