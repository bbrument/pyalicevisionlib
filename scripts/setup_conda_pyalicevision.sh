#!/bin/bash
# =============================================================================
# Setup script to add pyalicevision to your conda environment
# =============================================================================
# This script creates activation/deactivation hooks for your conda environment
# to automatically set up pyalicevision when you activate the environment.
#
# Usage:
#   ./setup_conda_pyalicevision.sh py_eval
#
# After running this script, pyalicevision will be available whenever you
# activate your conda environment:
#   conda activate py_eval
#   python -c "from pyalicevision import sfmData; print('OK')"
# =============================================================================

set -e

# Configuration - update these paths for your system
ALICEVISION_ROOT="/home/bbrument/dev/alicevision/AV_install"
ALICEVISION_DEPS_ROOT="/home/bbrument/dev/alicevision/AliceVisionDeps-2025.09.12-ubuntu22.04-cuda12.1.1"

# Get conda environment name from argument or use current
if [ -n "$1" ]; then
    ENV_NAME="$1"
    CONDA_PREFIX=$(conda info --envs | grep "^$ENV_NAME " | awk '{print $NF}')
    if [ -z "$CONDA_PREFIX" ]; then
        echo "Error: Conda environment '$ENV_NAME' not found"
        exit 1
    fi
else
    if [ -z "$CONDA_PREFIX" ]; then
        echo "Error: No conda environment active and no environment name provided"
        echo "Usage: $0 <conda_env_name>"
        exit 1
    fi
    ENV_NAME=$(basename "$CONDA_PREFIX")
fi

echo "Setting up pyalicevision for conda environment: $ENV_NAME"
echo "CONDA_PREFIX: $CONDA_PREFIX"

# Create directories for hooks
ACTIVATE_DIR="$CONDA_PREFIX/etc/conda/activate.d"
DEACTIVATE_DIR="$CONDA_PREFIX/etc/conda/deactivate.d"
mkdir -p "$ACTIVATE_DIR" "$DEACTIVATE_DIR"

# Create activation script
ACTIVATE_SCRIPT="$ACTIVATE_DIR/pyalicevision.sh"
cat > "$ACTIVATE_SCRIPT" << EOF
#!/bin/bash
# pyalicevision environment setup

export ALICEVISION_ROOT="$ALICEVISION_ROOT"
export ALICEVISION_DEPS_ROOT="$ALICEVISION_DEPS_ROOT"
export ALICEVISION_SENSOR_DB="\${ALICEVISION_ROOT}/share/aliceVision/cameraSensors.db"

# Save original values
export _PYAV_OLD_LD_LIBRARY_PATH="\${LD_LIBRARY_PATH:-}"
export _PYAV_OLD_PYTHONPATH="\${PYTHONPATH:-}"
export _PYAV_OLD_PATH="\${PATH:-}"

# Library paths
export LD_LIBRARY_PATH="\${ALICEVISION_DEPS_ROOT}/lib:\${ALICEVISION_ROOT}/lib:\${LD_LIBRARY_PATH}"

# Python path for pyalicevision
export PYTHONPATH="\${ALICEVISION_ROOT}/lib/python:\${ALICEVISION_ROOT}/lib/python3.11/site-packages:\${PYTHONPATH}"

# Executable path
export PATH="\${ALICEVISION_ROOT}/bin:\${PATH}"
EOF
chmod +x "$ACTIVATE_SCRIPT"

# Create deactivation script
DEACTIVATE_SCRIPT="$DEACTIVATE_DIR/pyalicevision.sh"
cat > "$DEACTIVATE_SCRIPT" << 'EOF'
#!/bin/bash
# pyalicevision environment cleanup

# Restore original values
if [ -n "${_PYAV_OLD_LD_LIBRARY_PATH+x}" ]; then
    export LD_LIBRARY_PATH="${_PYAV_OLD_LD_LIBRARY_PATH}"
    unset _PYAV_OLD_LD_LIBRARY_PATH
fi

if [ -n "${_PYAV_OLD_PYTHONPATH+x}" ]; then
    export PYTHONPATH="${_PYAV_OLD_PYTHONPATH}"
    unset _PYAV_OLD_PYTHONPATH
fi

if [ -n "${_PYAV_OLD_PATH+x}" ]; then
    export PATH="${_PYAV_OLD_PATH}"
    unset _PYAV_OLD_PATH
fi

unset ALICEVISION_ROOT
unset ALICEVISION_DEPS_ROOT
unset ALICEVISION_SENSOR_DB
EOF
chmod +x "$DEACTIVATE_SCRIPT"

echo ""
echo "✓ Created activation script: $ACTIVATE_SCRIPT"
echo "✓ Created deactivation script: $DEACTIVATE_SCRIPT"
echo ""
echo "To apply changes, reactivate your environment:"
echo "  conda deactivate && conda activate $ENV_NAME"
echo ""
echo "Then test pyalicevision:"
echo "  python -c \"from pyalicevision import sfmData; print('pyalicevision OK!')\""
