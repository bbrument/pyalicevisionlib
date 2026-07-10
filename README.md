# pyalicevisionlib

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Python utilities for working with **AliceVision/Meshroom** data:

- 📷 **Camera handling** — Load cameras from SfMData (JSON or via `pyalicevision`), project 3D points
- 📊 **Mesh evaluation** — Chamfer distance, precision/recall, F-score computation
- 🧹 **Visibility-based cleanup** — Remove invisible mesh regions using camera masks
- 🎨 **3D visualization** — Interactive camera and mesh visualization with matplotlib
- 🔄 **Format conversion** — RealityCapture (XMP + point clouds), COLMAP, Metashape, IDR ↔ AliceVision SfMData
- 🖼️ **Image processing** — Unified image I/O with EXR/HDR support, contour extraction

## Installation

### Basic installation

```bash
git clone https://github.com/bbrument/pyalicevisionlib.git
cd pyalicevisionlib
pip install -e .
```

This installs the library with JSON-only SfMData parsing.

### With pyalicevision support (optional)

`pyalicevision` is AliceVision's Python bindings, not available on PyPI.
See [docs/pyalicevision_setup.md](docs/pyalicevision_setup.md) for setup instructions.

With `pyalicevision`, you get:
- Support for `.sfm` and `.abc` binary formats
- Full SfMData structure normalization
- Access to AliceVision's native APIs

## Quick Start

### Python API

```python
from pyalicevisionlib import (
    load_sfmdata,
    load_cameras_from_sfmdata,
    Camera,
    evaluate_mesh,
)

# Load SfMData (works with .json, .sfm, .abc)
sfm = load_sfmdata("sfmdata.json")
cameras = sfm.get_cameras()

# Project 3D points to image coordinates
import numpy as np
points_3d = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
points_2d = cameras[0].project_points(points_3d)

# Get camera intrinsics matrix
K = cameras[0].get_K()

# Evaluate mesh reconstruction quality
from pyalicevisionlib.evaluation import run_evaluation
run_evaluation(
    data_mesh_path="reconstructed.ply",
    gt_mesh_path="ground_truth.ply",
    output_dir="results/",
    visualize=True
)
```

### CLI Tools

Full reference with all options: [docs/cli.md](docs/cli.md)

| Command | Purpose |
|---------|---------|
| `pyav-rc2sfm` / `pyav-sfm2rc` | RealityCapture XMP (+ point cloud) ↔ SfMData |
| `pyav-colmap2sfm` / `pyav-sfm2colmap` | COLMAP sparse reconstruction ↔ SfMData |
| `pyav-metashape2sfm` / `pyav-sfm2metashape` | Agisoft Metashape XML ↔ SfMData |
| `pyav-idr2sfm` / `pyav-sfm2idr` | IDR `cameras.npz` ↔ SfMData |
| `pyav-visualize` | 3D visualization of cameras and meshes |
| `pyav-evaluate` | Mesh quality evaluation against ground truth |
| `pyav-transform` | Apply a 4x4 transform to meshes and cameras |
| `pyav-contours` | Canny edge contours from SfMData images |
| `pyav-render-normals` | Render normal maps and masks from a mesh |

```bash
# Visualize cameras and mesh
pyav-visualize sfmdata.json --mesh mesh.ply --mesh-points 5000

# Evaluate mesh quality against ground truth
pyav-evaluate --data-mesh mesh.ply --gt-mesh gt.ply --output-dir results/

# Convert RealityCapture XMP (+ point cloud) to AliceVision SfMData
pyav-rc2sfm xmp_folder/ images_folder/ output.json --point-cloud cloud.ply

# Convert SfMData back to RealityCapture (XMP + point_cloud.ply)
pyav-sfm2rc sfmdata.json rc_export/

# Apply transformation to mesh and cameras
pyav-transform --mesh model.ply -t transform.npy -o aligned.ply \
               --cameras sfm.json --cameras-output sfm_aligned.json

# Extract Canny contours from images
pyav-contours --sfm sfmdata.json --masks masks/ --output contours/
```

## Features

### Camera Handling

```python
from pyalicevisionlib import Camera, load_cameras_from_sfmdata

cameras = load_cameras_from_sfmdata("sfmdata.json")

for cam in cameras:
    # Intrinsics
    K = cam.get_K()  # 3x3 intrinsic matrix
    P = cam.get_projection_matrix()  # 3x4 projection matrix
    
    # Extrinsics
    center = cam.center  # Camera position in world
    look_dir = cam.get_look_direction()  # Viewing direction
    
    # Projection
    uv = cam.project_points(points_3d)  # (N,3) -> (N,2)
```

### Mesh Evaluation

```python
from pyalicevisionlib import evaluate_mesh, ChamferResult, PrecisionRecallResult

# Basic evaluation
chamfer, pr = evaluate_mesh(
    data_mesh_path="reconstruction.ply",
    gt_mesh_path="ground_truth.ply",
    output_dir="results/",
    sampling_density=0.05
)

print(f"Chamfer distance: {chamfer.mean:.4f}")
print(f"Precision: {pr.precision:.2%}, Recall: {pr.recall:.2%}, F-score: {pr.f_score:.2%}")
```

### Visibility-based Cleanup

```python
from pyalicevisionlib import cleanup_mesh_visibility

# Remove mesh regions not visible from any camera
cleaned_mesh = cleanup_mesh_visibility(
    mesh_path="noisy_mesh.ply",
    cameras_json_path="sfmdata.json",
    output_path="cleaned_mesh.ply",
    use_alpha=True,  # Use alpha channel as masks
    dilation_radius=12
)
```

### Image I/O

```python
from pyalicevisionlib import load_image, load_gray, load_mask, save_image

# Unified loading (supports EXR, HDR, PNG, JPEG, etc.)
img = load_image("image.exr", mode='rgb', dtype='float32')
gray = load_gray("image.png", dtype='uint8')
mask = load_mask("mask.png")  # Returns boolean array

# Save with automatic format detection
save_image(img, "output.png")
```

## Coordinate Conventions

This library follows **AliceVision conventions**:

| Property | Convention |
|----------|------------|
| Rotation | `cam2world` (camera axes in world coordinates) |
| Center | Camera position in world coordinates |
| Principal Point | Offset from image center in pixels |
| World coordinates | Right-handed, Y-up |

**Projection formula:**
```
P_camera = R_world2cam @ (P_world - Center)
R_world2cam = R_cam2world.T
```

## Project Structure

```
pyalicevisionlib/
├── src/pyalicevisionlib/
│   ├── camera.py          # Camera class with projection
│   ├── sfmdata.py         # SfMData loading/saving
│   ├── mesh.py            # Mesh loading utilities
│   ├── image.py           # Unified image I/O
│   ├── utils.py           # Point cloud and transform utilities
│   ├── evaluation/        # Mesh evaluation metrics
│   │   ├── chamfer.py     # Chamfer distance
│   │   ├── cleanup.py     # Visibility-based cleanup
│   │   └── pipeline.py    # Full evaluation pipeline
│   ├── visualization/     # 3D plotting
│   └── scripts/           # CLI entry points
├── docs/                  # Additional documentation
├── pyproject.toml         # Package configuration
└── README.md
```

## Requirements

**Core dependencies:**
- numpy, scipy, trimesh, open3d, scikit-learn
- opencv-python, scikit-image
- matplotlib, tqdm

**Optional:**
- `pyalicevision` — For binary SfMData formats (.sfm, .abc)
- `OpenImageIO` — For professional image format support

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Lint
ruff check src/
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions welcome! Please open an issue or submit a pull request.

## Acknowledgments

Built for use with [AliceVision](https://alicevision.org/) and [Meshroom](https://alicevision.org/#meshroom).
