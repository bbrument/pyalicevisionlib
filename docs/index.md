# Project Documentation: pyalicevisionlib

**Generated:** 2026-01-23  
**Type:** Python Library  
**Version:** 0.1.0  
**Status:** Beta

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Technology Stack](#technology-stack)
3. [Architecture](#architecture)
4. [Module Structure](#module-structure)
5. [Key Features](#key-features)
6. [Dependencies](#dependencies)
7. [CLI Tools](#cli-tools)
8. [Development Setup](#development-setup)
9. [Coordinate Conventions](#coordinate-conventions)
10. [Integration Points](#integration-points)

---

## Project Overview

**pyalicevisionlib** is a Python utility library for working with AliceVision/Meshroom photogrammetry data. It provides comprehensive tools for camera handling, mesh evaluation, and 3D visualization.

### Purpose
- Simplify working with AliceVision SfMData structures
- Provide evaluation metrics for 3D reconstruction quality
- Enable visibility-based mesh cleanup
- Offer unified image I/O with EXR/HDR support
- Bridge between AliceVision and standard Python scientific stack

### Target Users
- Developers working with photogrammetry pipelines
- Researchers in computer vision and 3D reconstruction
- Scientists evaluating mesh reconstruction quality

### Project Classification
- **Type:** Library
- **Domain:** Computer Vision / 3D Reconstruction / Photogrammetry
- **Structure:** Monolithic
- **Language:** Python 3.9+

---

## Technology Stack

### Core Technologies

| Category | Technology | Version | Purpose |
|----------|-----------|---------|---------|
| **Language** | Python | 3.9+ | Primary language |
| **Build System** | Hatchling | latest | Package building |
| **Package Manager** | pip | latest | Dependency management |
| **Type Hints** | py.typed | - | Static type information |

### Key Dependencies

**Scientific Computing:**
- **numpy** (≥1.20) - Array operations, linear algebra
- **scipy** (≥1.7) - Scientific algorithms, spatial operations
- **scikit-learn** (≥1.0) - Nearest neighbor search, metrics

**3D Processing:**
- **trimesh** (≥3.0) - Mesh loading and manipulation
- **open3d** (≥0.15) - 3D data processing, point cloud operations
- **pyembree** (≥0.1.6) - Ray tracing acceleration

**Image Processing:**
- **opencv-python** (≥4.0) - Image I/O, transformations
- **scikit-image** (≥0.19) - Advanced image processing

**Visualization:**
- **matplotlib** (≥3.5) - 3D plotting, camera visualization

**Utilities:**
- **tqdm** (≥4.0) - Progress bars

### Optional Dependencies

**AliceVision Integration:**
- **pyalicevision** - Native AliceVision Python bindings
  - Supports .sfm and .abc binary formats
  - Full SfMData structure normalization
  - Not available on PyPI (must be built from source)

**Professional Image I/O:**
- **OpenImageIO** - Professional image format support

### Development Dependencies

| Tool | Version | Purpose |
|------|---------|---------|
| **pytest** | ≥7.0 | Unit testing |
| **pytest-cov** | ≥4.0 | Coverage reporting |
| **ruff** | ≥0.1 | Linting and formatting |
| **mypy** | ≥1.0 | Type checking |

---

## Architecture

### Architecture Pattern
**Modular Library Architecture** with clear separation of concerns:

```
┌─────────────────────────────────────────────┐
│            CLI Entry Points                  │
│  (visualize, evaluate, transform, etc.)     │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────▼──────────────────────────┐
│         Core Library Modules                 │
├─────────────────────────────────────────────┤
│  Camera  │  SfMData  │  Mesh  │  Image      │
│─────────────────────────────────────────────│
│  Evaluation  │  Visualization  │  Utils     │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────▼──────────────────────────┐
│       External Dependencies                  │
│  numpy, scipy, trimesh, open3d, opencv      │
└─────────────────────────────────────────────┘
                   │
         ┌─────────┴──────────┐
         ▼                    ▼
┌─────────────────┐  ┌──────────────────┐
│  AliceVision    │  │  Meshroom        │
│  (pyalicevision)│  │  Output Files    │
└─────────────────┘  └──────────────────┘
```

### Design Principles
1. **Unified Interfaces** - Single entry points for complex operations
2. **Backend Flexibility** - Support multiple data sources (JSON, pyalicevision)
3. **Numpy-centric** - Standard array operations throughout
4. **Fail-safe I/O** - Multiple fallback strategies for image loading
5. **Convention Consistency** - Strict adherence to AliceVision conventions

---

## Module Structure

### Package Layout

```
src/pyalicevisionlib/
├── __init__.py              # Main exports
├── py.typed                 # Type hint marker
│
├── camera.py                # Camera class and projection
├── sfmdata.py               # SfMData loading/wrapper
├── mesh.py                  # Mesh utilities
├── image.py                 # Unified image I/O
├── utils.py                 # Point cloud and transforms
│
├── evaluation/              # Mesh quality metrics
│   ├── __init__.py
│   ├── chamfer.py          # Chamfer distance computation
│   ├── cleanup.py          # Visibility-based cleanup
│   └── pipeline.py         # Full evaluation pipeline
│
├── visualization/           # 3D plotting
│   ├── __init__.py
│   └── plot.py             # Camera and mesh visualization
│
└── scripts/                 # CLI entry points
    ├── __init__.py
    ├── visualize.py        # pyav-visualize
    ├── evaluate.py         # pyav-evaluate
    ├── rc_to_sfmdata.py    # pyav-rc2sfm
    ├── transform.py        # pyav-transform
    └── contours.py         # pyav-contours
```

### Core Modules

#### camera.py
**Purpose:** Camera representation and projection operations

**Key Classes:**
- `Camera` - Camera with intrinsics, extrinsics, projection

**Key Functions:**
- `load_cameras_from_sfmdata()` - Extract cameras from SfMData
- `project_points()` - 3D to 2D projection
- `get_K()` - Intrinsic matrix
- `get_projection_matrix()` - Full 3x4 projection matrix

#### sfmdata.py
**Purpose:** SfMData loading with multiple backends

**Key Classes:**
- `SfMDataWrapper` - Unified interface for JSON/pyalicevision
- `load_sfmdata()` - Factory function for loading

**Features:**
- Automatic format detection (.json, .sfm, .abc)
- Graceful fallback to JSON-only
- Normalized data structure

#### mesh.py
**Purpose:** Mesh loading and point sampling

**Key Functions:**
- `load_mesh()` - Load mesh with trimesh
- `sample_points_on_mesh()` - Uniform point sampling

#### image.py
**Purpose:** Unified image I/O with multiple backends

**Key Functions:**
- `load_image()` - Load with OIIO/OpenCV/PIL fallback
- `load_gray()` - Grayscale loading
- `load_mask()` - Boolean mask loading
- `save_image()` - Format-aware saving

**Supported Formats:**
- Standard: PNG, JPEG, TIFF
- HDR: EXR, HDR, PFM

#### utils.py
**Purpose:** Point cloud operations and transformations

**Key Functions:**
- `filter_point_cloud_by_mesh()` - Spatial filtering
- `apply_transform()` - 4x4 transformation matrices
- `compute_bounding_box()` - Mesh bounds

### Evaluation Module

#### chamfer.py
**Purpose:** Chamfer distance computation

**Key Classes:**
- `ChamferResult` - Distance statistics
- `compute_chamfer_distance()` - Bidirectional distance

**Metrics:**
- Mean, median, std
- Min, max, percentiles
- Bidirectional averaging

#### cleanup.py
**Purpose:** Visibility-based mesh cleanup

**Key Functions:**
- `cleanup_mesh_visibility()` - Remove invisible regions
- Uses camera masks and ray tracing
- Morphological dilation for robustness

#### pipeline.py
**Purpose:** Complete evaluation workflow

**Key Functions:**
- `run_evaluation()` - Full precision/recall/Chamfer pipeline
- Automatic sampling
- Visualization generation

### Visualization Module

#### plot.py
**Purpose:** 3D camera and mesh visualization

**Key Functions:**
- `plot_cameras()` - Camera frustums in 3D
- `plot_mesh()` - Point cloud visualization
- `plot_cameras_and_mesh()` - Combined view

**Features:**
- matplotlib 3D axes
- Camera frustum rendering
- Point cloud sampling

---

## Key Features

### 1. Camera Handling

**Capabilities:**
- Load cameras from SfMData (JSON or pyalicevision)
- Project 3D points to image coordinates
- Extract intrinsic (K) and projection (P) matrices
- Compute camera centers and view directions
- World ↔ camera coordinate transformations

**Coordinate System:**
- AliceVision conventions (cam2world rotation)
- Right-handed, Y-up world coordinates
- Principal point as offset from image center

### 2. Mesh Evaluation

**Chamfer Distance:**
- Bidirectional symmetric distance
- Statistics: mean, median, std, percentiles
- Efficient nearest-neighbor search

**Precision/Recall:**
- Threshold-based metrics
- F-score computation
- Configurable distance thresholds

**Visibility Cleanup:**
- Remove mesh regions not visible from any camera
- Alpha channel mask support
- Morphological dilation for robustness

### 3. Format Conversion

**RealityCapture to AliceVision:**
- XMP camera data → SfMData JSON
- Preserves intrinsics and extrinsics
- Image path mapping

### 4. Visualization

**3D Plotting:**
- Camera frustums with look direction
- Mesh point clouds
- Interactive matplotlib views

### 5. Image I/O

**Unified Loading:**
- Automatic backend selection (OIIO → OpenCV → PIL)
- EXR/HDR support
- Type and colorspace conversion
- Mask loading (boolean arrays)

---

## Dependencies

### Direct Dependencies

```toml
[project.dependencies]
numpy>=1.20
scipy>=1.7
trimesh>=3.0
open3d>=0.15
scikit-learn>=1.0
matplotlib>=3.5
tqdm>=4.0
opencv-python>=4.0
scikit-image>=0.19
pyembree>=0.1.6
```

### Optional Dependencies

```toml
[project.optional-dependencies]
pyalicevision = []  # Not on PyPI - manual install
dev = [
    "pytest>=7.0",
    "pytest-cov>=4.0",
    "ruff>=0.1",
    "mypy>=1.0",
]
```

### Installation Methods

**Standard (JSON-only):**
```bash
pip install -e .
```

**With Development Tools:**
```bash
pip install -e ".[dev]"
```

**With pyalicevision:**
See `docs/pyalicevision_setup.md` for build instructions

---

## CLI Tools

### pyav-visualize
**Purpose:** Visualize cameras and meshes from SfMData

```bash
pyav-visualize sfmdata.json --mesh mesh.ply --mesh-points 5000
```

**Options:**
- `--mesh` - Mesh file to visualize
- `--mesh-points` - Number of points to sample
- Interactive 3D view

### pyav-evaluate
**Purpose:** Evaluate mesh reconstruction quality

```bash
pyav-evaluate --data-mesh mesh.ply --gt-mesh gt.ply --output-dir results/
```

**Outputs:**
- Chamfer distance statistics
- Precision/recall metrics
- Visualization plots

### pyav-rc2sfm
**Purpose:** Convert RealityCapture XMP to AliceVision SfMData

```bash
pyav-rc2sfm xmp_folder/ images_folder/ output.json
```

**Features:**
- Batch XMP processing
- Camera parameter conversion
- Image path mapping

### pyav-transform
**Purpose:** Apply transformations to meshes and cameras

```bash
pyav-transform --mesh model.ply -t transform.npy -o aligned.ply \
               --cameras sfm.json --cameras-output sfm_aligned.json
```

**Features:**
- 4x4 transformation matrices
- Mesh and camera transformation
- Alignment workflows

### pyav-contours
**Purpose:** Extract Canny edge contours from images

```bash
pyav-contours --sfm sfmdata.json --masks masks/ --output contours/
```

**Features:**
- Batch image processing
- Canny edge detection
- Contour extraction

---

## Development Setup

### Installation

```bash
# Clone repository
git clone https://github.com/bbrument/pyalicevisionlib.git
cd pyalicevisionlib

# Install with dev dependencies
pip install -e ".[dev]"
```

### Testing

```bash
# Run tests
pytest

# With coverage
pytest --cov=pyalicevisionlib
```

### Code Quality

```bash
# Linting
ruff check src/

# Type checking
mypy src/

# Formatting
ruff format src/
```

### Configuration

**ruff (tool.ruff):**
- Line length: 100
- Target: Python 3.9
- Select: E, F, I, W
- Ignore: E501

**mypy (tool.mypy):**
- Python 3.9
- Warn on unused configs
- Ignore missing imports

---

## Coordinate Conventions

### AliceVision Conventions

| Property | Convention |
|----------|------------|
| **Rotation** | `cam2world` - camera axes in world coordinates |
| **Center** | Camera position in world coordinates |
| **Principal Point** | Offset from image center in pixels |
| **World Coordinates** | Right-handed, Y-up |

### Projection Formula

```
P_camera = R_world2cam @ (P_world - Center)
R_world2cam = R_cam2world.T
```

### Camera Parameters

**Intrinsics:**
- Focal length (fx, fy) in pixels
- Principal point (cx, cy) as offset from center
- Distortion parameters (k1, k2, k3, p1, p2)

**Extrinsics:**
- Rotation matrix R (3x3) - cam2world
- Center (3,) - camera position in world

---

## Integration Points

### AliceVision Integration

**Input Formats:**
- SfMData JSON files from Meshroom
- Binary .sfm and .abc (with pyalicevision)
- Camera parameters from structure-from-motion

**Output Formats:**
- SfMData JSON compatible with Meshroom
- Camera transformations
- Cleaned meshes

### RealityCapture Integration

**Conversion:**
- XMP camera metadata → SfMData
- Maintains camera parameters
- Image path mapping

### Scientific Python Stack

**Data Flow:**
- numpy arrays for all numerical data
- trimesh for mesh representation
- open3d for point cloud operations
- matplotlib for visualization

### File Format Support

**Images:**
- PNG, JPEG, TIFF (via OpenCV/PIL)
- EXR, HDR (via OpenImageIO)
- PFM (via custom loader)

**Meshes:**
- PLY, OBJ, STL (via trimesh)
- Point clouds (numpy arrays)

**Data:**
- JSON (SfMData, transforms)
- numpy arrays (.npy)

---

## Existing Documentation

- **README.md** - Main project documentation
- **CHANGELOG.md** - Version history and changes
- **docs/pyalicevision_setup.md** - pyalicevision build guide

---

**Documentation Status:** Complete  
**Last Updated:** 2026-01-23  
**Scan Level:** Quick (pattern-based)
