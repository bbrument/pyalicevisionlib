"""
Point cloud I/O and conversion utilities.

Handles the point cloud formats produced by RealityCapture exports:
- PLY (ascii, binary_little_endian, binary_big_endian), with optional
  per-vertex colors and normals
- XYZ / TXT / CSV / PTS text files (x y z [r g b])

Also converts between raw point arrays and the AliceVision SfMData
'structure' section (landmarks), and between RealityCapture and
AliceVision world coordinates (same Y/Z flip as camera poses).

RealityCapture note:
    When exporting a point cloud intended to accompany XMP camera files,
    choose the "Same as XMP" coordinate system in the RC export dialog so
    that points and cameras share the same world frame.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from .sfmdata import WORLD_CORRECTION

# PLY property type -> numpy dtype (endianness prefix added at read time)
_PLY_DTYPES = {
    'char': 'i1', 'int8': 'i1',
    'uchar': 'u1', 'uint8': 'u1',
    'short': 'i2', 'int16': 'i2',
    'ushort': 'u2', 'uint16': 'u2',
    'int': 'i4', 'int32': 'i4',
    'uint': 'u4', 'uint32': 'u4',
    'float': 'f4', 'float32': 'f4',
    'double': 'f8', 'float64': 'f8',
}

_COLOR_PROPERTY_NAMES = {
    'red': 0, 'green': 1, 'blue': 2,
    'diffuse_red': 0, 'diffuse_green': 1, 'diffuse_blue': 2,
    'r': 0, 'g': 1, 'b': 2,
}

_TEXT_EXTENSIONS = {'.xyz', '.txt', '.csv', '.pts'}


# =============================================================================
# Loading
# =============================================================================

def load_point_cloud(path: Union[str, Path]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Load a point cloud from a PLY or text (xyz/txt/csv/pts) file.

    Args:
        path: Path to the point cloud file

    Returns:
        Tuple of (points, colors):
        - points: (N, 3) float64 array
        - colors: (N, 3) uint8 array, or None if the file has no colors

    Raises:
        FileNotFoundError: If the file does not exist
        ValueError: If the format is unsupported or the file is malformed
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Point cloud file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == '.ply':
        return _load_ply(path)
    if suffix in _TEXT_EXTENSIONS:
        return _load_text(path)

    raise ValueError(
        f"Unsupported point cloud format '{suffix}'. "
        f"Supported: .ply, {', '.join(sorted(_TEXT_EXTENSIONS))}"
    )


def _parse_ply_header(f) -> Tuple[str, List[Tuple[str, str]], int]:
    """
    Parse a PLY header from a binary file handle.

    Returns:
        Tuple of (format_name, vertex_properties, vertex_count) where
        vertex_properties is a list of (property_name, ply_type) pairs.

    Raises:
        ValueError: If the header is malformed or the vertex element is not
            the first element (variable-length elements cannot be skipped).
    """
    magic = f.readline().strip()
    if magic != b'ply':
        raise ValueError("Not a PLY file (missing 'ply' magic line)")

    fmt = None
    elements = []  # list of (name, count, [(prop_name, prop_type), ...])
    current = None

    while True:
        line = f.readline()
        if not line:
            raise ValueError("Unexpected end of file in PLY header")
        tokens = line.decode('ascii', errors='replace').strip().split()
        if not tokens or tokens[0] == 'comment' or tokens[0] == 'obj_info':
            continue
        if tokens[0] == 'format':
            fmt = tokens[1]
        elif tokens[0] == 'element':
            current = (tokens[1], int(tokens[2]), [])
            elements.append(current)
        elif tokens[0] == 'property':
            if current is None:
                raise ValueError("PLY property declared before any element")
            if tokens[1] == 'list':
                current[2].append((tokens[-1], 'list'))
            else:
                current[2].append((tokens[-1], tokens[1]))
        elif tokens[0] == 'end_header':
            break

    if fmt not in ('ascii', 'binary_little_endian', 'binary_big_endian'):
        raise ValueError(f"Unsupported PLY format: {fmt}")

    vertex_elements = [e for e in elements if e[0] == 'vertex']
    if not vertex_elements:
        raise ValueError("PLY file has no 'vertex' element")
    if elements[0][0] != 'vertex':
        raise ValueError("PLY 'vertex' element must be the first element")

    name, count, props = elements[0]
    if any(ptype == 'list' for _, ptype in props):
        raise ValueError("PLY vertex element with list properties is not supported")

    return fmt, props, count


def _extract_points_and_colors(
    data: Dict[str, np.ndarray]
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Build (points, colors) arrays from per-property vertex columns."""
    for axis in ('x', 'y', 'z'):
        if axis not in data:
            raise ValueError(f"PLY vertex element is missing '{axis}' property")

    points = np.column_stack([
        data['x'].astype(np.float64),
        data['y'].astype(np.float64),
        data['z'].astype(np.float64),
    ])

    color_columns = {}
    for prop_name, channel in _COLOR_PROPERTY_NAMES.items():
        if prop_name in data and channel not in color_columns:
            color_columns[channel] = data[prop_name]

    colors = None
    if len(color_columns) == 3:
        stacked = np.column_stack([color_columns[c] for c in (0, 1, 2)])
        if stacked.dtype.kind == 'f':
            # Float colors are assumed normalized in [0, 1]
            stacked = np.clip(stacked * 255.0, 0, 255)
        colors = stacked.astype(np.uint8)

    return points, colors


def _load_ply(path: Path) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Load vertices (and colors if present) from a PLY file."""
    with open(path, 'rb') as f:
        fmt, props, count = _parse_ply_header(f)

        for _, ptype in props:
            if ptype not in _PLY_DTYPES:
                raise ValueError(f"Unsupported PLY property type: {ptype}")

        if fmt == 'ascii':
            rows = []
            while len(rows) < count:
                line = f.readline()
                if not line:
                    raise ValueError("Unexpected end of file in PLY vertex data")
                tokens = line.split()
                if tokens:
                    rows.append([float(t) for t in tokens[:len(props)]])
            raw = np.array(rows, dtype=np.float64)
            data = {
                name: raw[:, i].astype(_PLY_DTYPES[ptype])
                for i, (name, ptype) in enumerate(props)
            }
        else:
            endian = '<' if fmt == 'binary_little_endian' else '>'
            dtype = np.dtype([
                (name, endian + _PLY_DTYPES[ptype]) for name, ptype in props
            ])
            buffer = f.read(dtype.itemsize * count)
            if len(buffer) < dtype.itemsize * count:
                raise ValueError("Unexpected end of file in PLY vertex data")
            raw = np.frombuffer(buffer, dtype=dtype, count=count)
            data = {name: raw[name] for name, _ in props}

    return _extract_points_and_colors(data)


def _load_text(path: Path) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Load a text point cloud (xyz/txt/csv/pts): 'x y z [r g b]' per line.

    Colors are detected as three trailing integer-valued columns in [0, 255]
    with at least one value above 1 (to distinguish them from normals).
    """
    try:
        data = np.loadtxt(str(path), delimiter=None, ndmin=2)
    except ValueError:
        data = np.loadtxt(str(path), delimiter=',', ndmin=2)

    if data.size == 0:
        raise ValueError(f"Empty point cloud file: {path}")
    if data.shape[1] < 3:
        raise ValueError(
            f"Text point cloud must have at least 3 columns, got {data.shape[1]}"
        )

    points = data[:, :3].astype(np.float64)

    colors = None
    if data.shape[1] >= 6:
        candidate = data[:, -3:]
        is_integer = np.allclose(candidate, np.round(candidate))
        in_range = candidate.min() >= 0 and candidate.max() <= 255
        if is_integer and in_range and candidate.max() > 1:
            colors = candidate.astype(np.uint8)

    return points, colors


# =============================================================================
# Saving
# =============================================================================

def save_point_cloud_ply(
    path: Union[str, Path],
    points: np.ndarray,
    colors: Optional[np.ndarray] = None,
    binary: bool = True,
) -> None:
    """
    Save a point cloud as a PLY file.

    Args:
        path: Output PLY path
        points: (N, 3) array of point positions
        colors: Optional (N, 3) uint8 array of RGB colors
        binary: Write binary_little_endian (default) or ascii

    Raises:
        ValueError: If points is empty or shapes are inconsistent
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
        raise ValueError(f"points must be a non-empty (N, 3) array, got {points.shape}")
    if colors is not None:
        colors = np.asarray(colors)
        if colors.shape != points.shape:
            raise ValueError(
                f"colors shape {colors.shape} does not match points shape {points.shape}"
            )
        colors = colors.astype(np.uint8)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    header_lines = [
        'ply',
        'format binary_little_endian 1.0' if binary else 'format ascii 1.0',
        f'element vertex {len(points)}',
        'property float x',
        'property float y',
        'property float z',
    ]
    if colors is not None:
        header_lines += [
            'property uchar red',
            'property uchar green',
            'property uchar blue',
        ]
    header_lines.append('end_header')
    header = '\n'.join(header_lines) + '\n'

    if binary:
        fields = [('x', '<f4'), ('y', '<f4'), ('z', '<f4')]
        if colors is not None:
            fields += [('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
        records = np.empty(len(points), dtype=np.dtype(fields))
        records['x'], records['y'], records['z'] = points.T.astype(np.float32)
        if colors is not None:
            records['red'], records['green'], records['blue'] = colors.T
        with open(path, 'wb') as f:
            f.write(header.encode('ascii'))
            f.write(records.tobytes())
    else:
        with open(path, 'w', encoding='ascii') as f:
            f.write(header)
            for i in range(len(points)):
                row = ' '.join(f'{v:.9g}' for v in points[i])
                if colors is not None:
                    row += ' ' + ' '.join(str(int(c)) for c in colors[i])
                f.write(row + '\n')


# =============================================================================
# Coordinate conversion (RealityCapture <-> AliceVision)
# =============================================================================

def rc_points_to_av(points: np.ndarray) -> np.ndarray:
    """
    Convert 3D points from RealityCapture world to AliceVision world.

    Applies the same Y/Z flip (WORLD_CORRECTION) that is applied to camera
    poses in rc_to_sfmdata: X_av = WORLD_CORRECTION @ X_rc.

    Args:
        points: (N, 3) array in RC coordinates

    Returns:
        New (N, 3) array in AliceVision coordinates
    """
    return np.asarray(points, dtype=np.float64) @ WORLD_CORRECTION.T


def av_points_to_rc(points: np.ndarray) -> np.ndarray:
    """
    Convert 3D points from AliceVision world to RealityCapture world.

    WORLD_CORRECTION is its own inverse, so this is the same flip as
    rc_points_to_av.

    Args:
        points: (N, 3) array in AliceVision coordinates

    Returns:
        New (N, 3) array in RC coordinates
    """
    return rc_points_to_av(points)


# =============================================================================
# SfMData 'structure' (landmarks) conversion
# =============================================================================

def landmarks_from_points(
    points: np.ndarray,
    colors: Optional[np.ndarray] = None,
    desc_type: str = 'sift',
) -> List[Dict]:
    """
    Convert a point array to an SfMData 'structure' landmark list.

    Landmarks are created without observations (positions only), which is
    sufficient for visualization and downstream geometry nodes.

    Args:
        points: (N, 3) array of point positions (AliceVision world)
        colors: Optional (N, 3) uint8 RGB array (defaults to white)
        desc_type: AliceVision describer type stored in each landmark

    Returns:
        List of landmark dicts matching the AliceVision JSON format
    """
    points = np.asarray(points, dtype=np.float64)
    structure = []
    for i in range(len(points)):
        color = colors[i] if colors is not None else (255, 255, 255)
        structure.append({
            'landmarkId': str(i),
            'descType': desc_type,
            'color': [str(int(c)) for c in color],
            'X': [str(v) for v in points[i]],
            'observations': [],
        })
    return structure


def points_from_landmarks(structure: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract points and colors from an SfMData 'structure' landmark list.

    Args:
        structure: List of landmark dicts (AliceVision JSON format)

    Returns:
        Tuple of (points, colors):
        - points: (N, 3) float64 array
        - colors: (N, 3) uint8 array (white for landmarks without color)
    """
    if not structure:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.uint8)

    points = np.array([
        [float(v) for v in lm['X']] for lm in structure
    ], dtype=np.float64)
    colors = np.array([
        [int(float(v)) for v in lm.get('color', ['255', '255', '255'])]
        for lm in structure
    ], dtype=np.uint8)
    return points, colors
