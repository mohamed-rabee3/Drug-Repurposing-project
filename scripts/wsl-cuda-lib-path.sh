#!/usr/bin/env bash
# Source after activating your venv so PyTorch's bundled CUDA libs (e.g. libnvrtc.so.12)
# are visible when loading DGL Graphbolt. Usage:
#   source venv/bin/activate
#   source scripts/wsl-cuda-lib-path.sh
#   python -c "import dgl"

set -e
TORCH_LIB="$(python -c 'import os, torch; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')"
export LD_LIBRARY_PATH="${TORCH_LIB}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
echo "LD_LIBRARY_PATH prepended with: $TORCH_LIB"
