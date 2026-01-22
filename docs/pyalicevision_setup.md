# Setting up pyalicevision

`pyalicevision` is AliceVision's Python bindings. It's not available on PyPI and must be 
built from source or obtained from a Meshroom installation.

## Option 1: Use Meshroom's setup script (Recommended)

If you have a Meshroom development environment:

```bash
# Source the Meshroom environment before running Python
source /path/to/Meshroom/setup_env.sh
python your_script.py
```

### Creating a wrapper script

Create a script to easily run Python with pyalicevision:

```bash
#!/bin/bash
# run_with_pyav.sh
source /path/to/Meshroom/setup_env.sh
python "$@"
```

Usage:
```bash
./run_with_pyav.sh your_script.py
```

## Option 2: Set environment variables manually

Add to your `.bashrc` or activate script:

```bash
export ALICEVISION_ROOT="/path/to/AV_install"
export ALICEVISION_DEPS_ROOT="/path/to/AliceVisionDeps"

# Library path for C++ dependencies
export LD_LIBRARY_PATH="${ALICEVISION_DEPS_ROOT}/lib:${ALICEVISION_ROOT}/lib:${LD_LIBRARY_PATH}"

# Python path for pyalicevision
export PYTHONPATH="${ALICEVISION_ROOT}/lib/python:${ALICEVISION_ROOT}/lib/python3.11/site-packages:${PYTHONPATH}"
```

## Option 3: Conda environment integration

If you use conda (e.g., `py_eval` environment), you can create an activation script:

```bash
# Create activation script
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
cat > $CONDA_PREFIX/etc/conda/activate.d/pyalicevision.sh << 'EOF'
#!/bin/bash
export ALICEVISION_ROOT="/home/bbrument/dev/alicevision/AV_install"
export ALICEVISION_DEPS_ROOT="/home/bbrument/dev/alicevision/AliceVisionDeps-2025.09.12-ubuntu22.04-cuda12.1.1"
export LD_LIBRARY_PATH="${ALICEVISION_DEPS_ROOT}/lib:${ALICEVISION_ROOT}/lib:${LD_LIBRARY_PATH}"
export PYTHONPATH="${ALICEVISION_ROOT}/lib/python:${ALICEVISION_ROOT}/lib/python3.11/site-packages:${PYTHONPATH}"
EOF

# Create deactivation script
mkdir -p $CONDA_PREFIX/etc/conda/deactivate.d
cat > $CONDA_PREFIX/etc/conda/deactivate.d/pyalicevision.sh << 'EOF'
#!/bin/bash
# Reset variables (optional - they'll be overwritten on next activation anyway)
unset ALICEVISION_ROOT
unset ALICEVISION_DEPS_ROOT
EOF

chmod +x $CONDA_PREFIX/etc/conda/activate.d/pyalicevision.sh
chmod +x $CONDA_PREFIX/etc/conda/deactivate.d/pyalicevision.sh
```

Then reactivate your environment:
```bash
conda deactivate
conda activate py_eval
```

## Verifying Installation

```python
try:
    from pyalicevision import sfmData, sfmDataIO
    print("✓ pyalicevision available")
except ImportError as e:
    print(f"✗ pyalicevision not available: {e}")
```

## Fallback Mode

`pyalicevisionlib` works without `pyalicevision` by parsing SfMData JSON files directly.
You'll see a warning at import:

```
Warning: pyalicevision not found. Using JSON fallback.
```

This is fine for most use cases. `pyalicevision` is only required for:
- Reading binary SfMData formats (`.sfm`, `.abc`)
- Advanced intrinsics (distortion models)
- Writing SfMData files
