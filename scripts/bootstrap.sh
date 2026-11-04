#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
CONDA_BIN="${CONDA_BIN:-/home/chishan/miniconda3/bin/conda}"
if ! "$CONDA_BIN" env list | awk '{print $1}' | grep -qx nano-interpretability; then
  "$CONDA_BIN" create -y -n nano-interpretability --file conda-linux-64.lock
fi
"$CONDA_BIN" env config vars set -n nano-interpretability PYTHONPATH= PYTHONNOUSERSITE=1
"$CONDA_BIN" run --no-capture-output -n nano-interpretability python -m pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu128 -r requirements.lock
"$CONDA_BIN" run --no-capture-output -n nano-interpretability python -m pip install --no-deps --no-build-isolation -e .
"$CONDA_BIN" run --no-capture-output -n nano-interpretability nanoinspect doctor
