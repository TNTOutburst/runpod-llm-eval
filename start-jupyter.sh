#!/usr/bin/env bash
set -euo pipefail

mkdir -p /workspace
cd /workspace

: "${JUPYTER_TOKEN:?Set JUPYTER_TOKEN in the RunPod template}"

echo "=== Installed inference stack ==="

python3 - <<'PY'
import sys
from importlib.metadata import version

print("Python:", sys.version)

for package in (
    "vllm",
    "torch",
    "transformers",
    "flashinfer-python",
    "huggingface-hub",
):
    try:
        print(f"{package}: {version(package)}")
    except Exception:
        pass
PY

echo "=== Starting JupyterLab on port 8888 ==="

exec python3 -m jupyterlab \
    --ServerApp.root_dir=/workspace \
    --ServerApp.ip=0.0.0.0 \
    --ServerApp.port=8888 \
    --ServerApp.open_browser=False \
    --ServerApp.allow_remote_access=True \
    --allow-root
