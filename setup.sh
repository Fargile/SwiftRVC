#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${ROOT_DIR}"

# python venv
PYTHON_BIN="${PYTHON_BIN:-python3}"
if command -v python3.12 >/dev/null 2>&1; then
    PYTHON_BIN="python3.12"
elif command -v python3.11 >/dev/null 2>&1; then
    PYTHON_BIN="python3.11"
fi

if [ ! -d ".venv" ]; then
    echo "creating .venv with ${PYTHON_BIN}..."
    "${PYTHON_BIN}" -m venv .venv
fi

.venv/bin/pip install --upgrade pip >/dev/null
.venv/bin/pip install -r requirements.txt
	
    
  
# mlx base weights
mkdir -p base_models models beatrice_models recordings

if [ ! -f "base_models/contentvec_base.safetensors" ]; then
    echo "downloading contentvec_base.safetensors..."
    curl -L --fail --progress-bar \
        "https://huggingface.co/lexandstuff/mlx-contentvec/resolve/main/contentvec_base.safetensors" \
        -o "base_models/contentvec_base.safetensors"
fi

if [ ! -f "base_models/rmvpe.safetensors" ]; then
    echo "downloading rmvpe.safetensors..."
    curl -L --fail --progress-bar \
        "https://huggingface.co/lexandstuff/mlx-rmvpe/resolve/main/rmvpe.safetensors" \
        -o "base_models/rmvpe.safetensors"
fi
	
    
# optional beatrice bridge
if [ -f "beatrice/lib/libbeatrice.a" ] && [ -f "beatrice/lib/beatrice.h" ]; then
    echo "building beatrice/libbeatrice_bridge.dylib..."
    clang++ -std=c++17 -O3 -arch arm64 -dynamiclib \
        -I"${ROOT_DIR}/beatrice/lib" \
        "${ROOT_DIR}/beatrice/beatrice_bridge.cpp" \
        "${ROOT_DIR}/beatrice/lib/libbeatrice.a" \
        -o "${ROOT_DIR}/beatrice/libbeatrice_bridge.dylib"
else
    echo "beatrice optional: drop arm64 libbeatrice.a + beatrice.h into beatrice/lib/ and re-run ./setup.sh"
fi

echo "setup done. run ./build.sh then open SwiftRVC.app"
