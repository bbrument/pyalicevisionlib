# CLI Reference

All command-line tools installed with `pyalicevisionlib`. Every tool also accepts `--help`.

| Command | Purpose |
|---------|---------|
| [`pyav-rc2sfm`](#pyav-rc2sfm) | RealityCapture XMP (+ point cloud) → SfMData |
| [`pyav-sfm2rc`](#pyav-sfm2rc) | SfMData → RealityCapture XMP (+ point cloud) |
| [`pyav-colmap2sfm`](#pyav-colmap2sfm) | COLMAP sparse reconstruction → SfMData |
| [`pyav-sfm2colmap`](#pyav-sfm2colmap) | SfMData → COLMAP text format |
| [`pyav-metashape2sfm`](#pyav-metashape2sfm) | Agisoft Metashape XML → SfMData |
| [`pyav-sfm2metashape`](#pyav-sfm2metashape) | SfMData → Agisoft Metashape XML |
| [`pyav-idr2sfm`](#pyav-idr2sfm) | IDR `cameras.npz` → SfMData |
| [`pyav-sfm2idr`](#pyav-sfm2idr) | SfMData → IDR `cameras.npz` |
| [`pyav-visualize`](#pyav-visualize) | 3D visualization of cameras and meshes |
| [`pyav-evaluate`](#pyav-evaluate) | Mesh quality evaluation against ground truth |
| [`pyav-transform`](#pyav-transform) | Apply a 4x4 transform to meshes and cameras |
| [`pyav-contours`](#pyav-contours) | Canny edge contours from SfMData images |
| [`pyav-render-normals`](#pyav-render-normals) | Render normal maps and masks from a mesh |

Conversion tools handle coordinate conventions automatically (AliceVision uses
cam2world rotations and a Y/Z axis flip relative to most other tools). See
[Coordinate Conventions](#coordinate-conventions) at the bottom of this page.

---

## Format conversion

### pyav-rc2sfm

Convert RealityCapture XMP camera files to AliceVision SfMData.

```bash
pyav-rc2sfm <xmp_folder> <images_folder> <output.json> [options]
```

| Argument / Option | Description |
|-------------------|-------------|
| `xmp_folder` | Folder containing RealityCapture XMP files (one per image) |
| `images_folder` | Folder containing the source images |
| `output` | Output SfMData JSON path |
| `--reference, -r` | Reference SfMData to match viewIds by image name |
| `--point-cloud, -p` | Point cloud exported from RealityCapture (`.ply`, `.xyz`, `.txt`, `.csv`, `.pts`); imported as SfMData landmarks (`structure`) |
| `--sensor-width` | Sensor width in mm (default: 36.0) |
| `--sensor-height` | Sensor height in mm (default: 24.0) |
| `--camera-make` / `--camera-model` / `--serial-number` | Camera metadata stored in the output |

```bash
pyav-rc2sfm ./xmp_files ./images output.json
pyav-rc2sfm ./xmp_files ./images output.json --sensor-width 23.5
pyav-rc2sfm ./xmp_files ./images output.json --reference ref_sfmdata.json
pyav-rc2sfm ./xmp_files ./images output.json --point-cloud cloud.ply
```

**Point cloud note:** in RealityCapture, export the point cloud with the
**"Same as XMP"** coordinate system so that points and XMP cameras share the
same world frame. The converter applies the RC → AliceVision axis correction to
both cameras and points.

### pyav-sfm2rc

Convert AliceVision SfMData to RealityCapture XMP camera files. If the SfMData
contains landmarks (`structure`), they are exported as `point_cloud.ply` next
to the XMP files, in the same (XMP) coordinate frame.

```bash
pyav-sfm2rc <sfmdata> <output_folder> [options]
```

| Argument / Option | Description |
|-------------------|-------------|
| `sfmdata` | Input SfMData file (`.json`, `.sfm`, `.abc`) |
| `output_folder` | Output folder for XMP files, images and point cloud |
| `--images-folder, -i` | Name of the images subfolder (default: `images`) |
| `--no-copy-images` | Do not copy images to the output folder |
| `--sensor-width` | Default sensor width in mm (default: 36.0) |
| `--no-point-cloud` | Do not export the SfMData landmarks as a PLY point cloud |
| `--point-cloud-filename` | Name of the exported PLY (default: `point_cloud.ply`) |

```bash
pyav-sfm2rc sfmdata.json ./output
pyav-sfm2rc sfmdata.json ./output --images-folder photos
pyav-sfm2rc sfmdata.json ./output --no-copy-images --no-point-cloud
```

### pyav-colmap2sfm

Convert a COLMAP sparse reconstruction (text model) to AliceVision SfMData.

```bash
pyav-colmap2sfm <colmap_dir> <output.json> [options]
```

| Argument / Option | Description |
|-------------------|-------------|
| `colmap_dir` | COLMAP text model directory (`cameras.txt`, `images.txt`) |
| `output` | Output SfMData JSON path |
| `--sensor-width` | Sensor width in mm (default: 36.0) |
| `--images-dir` | Directory containing the actual image files |
| `--reference, -r` | Reference SfMData to match viewIds and extract metadata |

```bash
pyav-colmap2sfm ./sparse_txt/0 output.json --sensor-width 36.0 --images-dir ./images
```

### pyav-sfm2colmap

Convert AliceVision SfMData to COLMAP text format.

```bash
pyav-sfm2colmap <sfmdata> <output_dir> [--sensor-width MM]
```

| Argument / Option | Description |
|-------------------|-------------|
| `sfmdata` | Input SfMData JSON path |
| `output_dir` | Output directory for COLMAP text files |
| `--sensor-width` | Override sensor width in mm (uses the SfMData value if not specified) |

```bash
pyav-sfm2colmap sfmdata.json ./colmap_output
```

### pyav-metashape2sfm

Convert an Agisoft Metashape camera XML export to AliceVision SfMData.

```bash
pyav-metashape2sfm <metashape.xml> <output.json> [options]
```

| Argument / Option | Description |
|-------------------|-------------|
| `xml_path` | Input Metashape XML file |
| `output` | Output SfMData JSON file |
| `--images-folder, -i` | Folder containing images |
| `--sensor-width` | Sensor width in mm (default: 36.0) |
| `--sensor-height` | Sensor height in mm (default: 24.0) |
| `--reference, -r` | Reference SfMData to match viewIds by image name |

```bash
pyav-metashape2sfm metashape.xml output.json --images-folder ./images
```

### pyav-sfm2metashape

Convert AliceVision SfMData to an Agisoft Metashape camera XML.

```bash
pyav-sfm2metashape <sfmdata> <output.xml> [--sensor-width MM]
```

| Argument / Option | Description |
|-------------------|-------------|
| `sfmdata` | Input SfMData file (`.json`, `.sfm`, `.abc`) |
| `output` | Output Metashape XML file |
| `--sensor-width` | Sensor width in mm (default: 36.0) |

```bash
pyav-sfm2metashape sfmdata.json metashape.xml
```

### pyav-idr2sfm

Convert an IDR-style `cameras.npz` (world matrices + scale matrices) to
AliceVision SfMData.

```bash
pyav-idr2sfm <cameras.npz> <output.json> [options]
```

| Argument / Option | Description |
|-------------------|-------------|
| `npz_path` | Path to `cameras.npz` |
| `output` | Output SfMData JSON path |
| `--images-folder, -i` | Folder with source images |
| `--sensor-width` | Sensor width in mm (default: 36.0) |
| `--no-save-scale-mats` | Disable saving the `scale_mats.npz` sidecar |

```bash
pyav-idr2sfm cameras.npz output.json -i /path/to/images
```

### pyav-sfm2idr

Convert AliceVision SfMData to an IDR dataset (`cameras.npz`).

```bash
pyav-sfm2idr <sfmdata> <output_folder> [options]
```

| Argument / Option | Description |
|-------------------|-------------|
| `sfmdata` | Input SfMData file |
| `output_folder` | Output IDR directory |
| `--scale-mode` | Scale matrix computation: `identity` (default), `masks`, `mesh`, `file` |
| `--masks-folder` | Path to masks (for `--scale-mode masks`, also copied to output) |
| `--geometry` | Path to mesh/point cloud (for `--scale-mode mesh`) |
| `--scale-mats` | Path to an existing `scale_mats.npz` (for `--scale-mode file`) |

```bash
pyav-sfm2idr sfmdata.json output_dir --scale-mode masks --masks-folder /path/to/masks
pyav-sfm2idr sfmdata.json output_dir --scale-mode mesh --geometry mesh.ply
```

---

## Visualization and evaluation

### pyav-visualize

Interactive 3D visualization of cameras (and optionally a mesh) from one or
more SfMData files.

```bash
pyav-visualize <sfmdata_files...> [options]
```

| Argument / Option | Description |
|-------------------|-------------|
| `sfmdata_files` | One or more SfMData files |
| `--labels` | Labels for each file |
| `--mesh` | Mesh file to display |
| `--mesh-points` | Max mesh points to display |
| `--axis-scale` | Camera axis scale |
| `--no-axes` | Hide camera coordinate axes |
| `--print-intrinsics` | Print intrinsics info |
| `--title` | Plot title |

```bash
pyav-visualize sfmdata.json --mesh mesh.ply --mesh-points 5000
pyav-visualize cameras1.json cameras2.json --labels "Set 1" "Set 2"
```

### pyav-evaluate

Evaluate a reconstructed mesh against a ground-truth mesh (Chamfer distance,
precision/recall, F-score), with optional visibility-based cleanup.

```bash
pyav-evaluate --data-mesh <mesh> --gt-mesh <gt> --output-dir <dir> [options]
```

| Option | Description |
|--------|-------------|
| `--data-mesh` | Path to the reconstructed mesh (`.ply`, `.obj`, ...) |
| `--gt-mesh` | Path to the ground-truth mesh |
| `--output-dir` | Directory for output files |
| `--cameras-masks` | SfMData JSON with mask images; required for `--cleanup` |
| `--cleanup` | Perform visibility-based mesh cleanup before evaluation |
| `--dilation-radius` | Mask dilation radius in pixels (default: 12) |
| `--z-threshold` | Minimum z-coordinate threshold for filtering |
| `--sampling-density` | Point cloud sampling density (default: 0.05) |
| `--max-dist` | Maximum distance for outlier filtering (default: 2.0) |
| `--visualize` | Generate error visualization point clouds |
| `--vis-threshold` | Error colormap threshold (default: 1.0) |
| `--quiet` | Suppress progress messages |

```bash
pyav-evaluate --data-mesh mesh.ply --gt-mesh gt.ply --output-dir results/
pyav-evaluate --data-mesh mesh.ply --gt-mesh gt.ply \
    --cameras-masks sfm.json --output-dir results/ --cleanup --visualize
```

---

## Geometry and image processing

### pyav-transform

Apply a 4x4 transformation matrix to a mesh and/or camera poses (projections
remain consistent when both are transformed together).

```bash
pyav-transform --transform <T.npy|T.txt> [options]
```

| Option | Description |
|--------|-------------|
| `--transform, -t` | 4x4 transformation matrix (`.npy` or `.txt`) — required |
| `--mesh, -m` | Input mesh path |
| `--output, -o` | Output mesh path |
| `--inverse, -i` | Apply the inverse transformation |
| `--cameras, -c` | Input SfMData JSON path |
| `--cameras-output` | Output SfMData JSON path |

```bash
pyav-transform --mesh model.ply -t transform.npy -o model_aligned.ply
pyav-transform --mesh model.ply -t transform.npy -o model_aligned.ply \
               --cameras sfm.json --cameras-output sfm_aligned.json
```

### pyav-contours

Generate Canny edge contours from SfMData images, restricted by masks.

```bash
pyav-contours --sfm <sfmdata.json> --masks <folder> --output <folder> [options]
```

| Option | Description |
|--------|-------------|
| `--sfm, -s` | SfMData JSON file |
| `--masks, -m` | Masks folder |
| `--output, -o` | Output folder |
| `--use-alpha` | Extract mask from the alpha channel |
| `--margin-size` | Margin size in pixels (default: 20) |
| `--canny-low` / `--canny-high` | Canny thresholds (default: 50 / 150) |
| `--sobel-kernel` | Sobel kernel size (default: 3) |
| `--component-mode` | Connected component filter: `center_point`, `smallest_area`, `hybrid` (default) |
| `--num-workers` | Number of parallel workers |
| `--force` | Regenerate existing outputs |
| `--image-filter` | Only process views whose filename contains this substring |
| `--save-masks` | Save filtered mask PNGs to this folder |

```bash
pyav-contours --sfm sfmdata.json --masks masks/ --output contours/ --use-alpha
```

### pyav-render-normals

Render per-view normal maps and masks from a mesh and SfMData cameras
(ray-traced, with anti-aliasing).

```bash
pyav-render-normals <sfmdata> <mesh> -o <output_dir> [options]
```

| Argument / Option | Description |
|-------------------|-------------|
| `sfmdata` | SfMData file path |
| `mesh` | Mesh file path |
| `--output, -o` | Output directory — required |
| `--views` | Specific view IDs to render (default: all) |
| `--chunk-size` | Rays per batch (default: 1000000) |
| `--samples` | Sub-pixel grid size for anti-aliasing (default: 3 = 3x3) |

```bash
pyav-render-normals sfmdata.json mesh.ply -o output/ --views 123456 789012
```

---

## Coordinate Conventions

AliceVision/Meshroom conventions used by all tools:

| Property | Convention |
|----------|------------|
| Rotation | `cam2world` (camera axes in world coordinates) |
| Center | Camera position in world coordinates |
| Principal point | Offset from image center in pixels |
| World correction | `diag(1, -1, -1)` Y/Z flip applied when converting from/to external tools (RealityCapture, COLMAP, Metashape, IDR) |

The same world correction is applied to camera poses **and** 3D point clouds
(landmarks), so cameras and points stay consistent across conversions.
