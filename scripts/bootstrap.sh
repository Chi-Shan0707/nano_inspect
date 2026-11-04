#!/usr/bin/env bash
# Create (or update) the dedicated conda environment from the pinned locks.
#   ENV_NAME       environment name (default nano-interpretability)
#   PIP_ARGS       extra pip flags (default --no-cache-dir)
#   DOCTOR_OUTPUT  where `nanoinspect doctor` writes its report (default runs/doctor.json)
set -euo pipefail
cd "$(dirname "$0")/.."
CONDA_BIN="${CONDA_BIN:-/home/chishan/miniconda3/bin/conda}"
ENV_NAME="${ENV_NAME:-nano-interpretability}"
PIP_ARGS="${PIP_ARGS:---no-cache-dir}"
if ! "$CONDA_BIN" env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  # The explicit lock references only conda-forge; never consult the default channels.
  "$CONDA_BIN" create -y -n "$ENV_NAME" --override-channels -c conda-forge --file conda-linux-64.lock
fi
"$CONDA_BIN" env config vars set -n "$ENV_NAME" PYTHONPATH= PYTHONNOUSERSITE=1
"$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" python -m pip install $PIP_ARGS --extra-index-url https://download.pytorch.org/whl/cu128 -r requirements.lock
"$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" python -m pip install --no-deps --no-build-isolation -e .
"$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" nanoinspect doctor --output "${DOCTOR_OUTPUT:-runs/doctor.json}"
