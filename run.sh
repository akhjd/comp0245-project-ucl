#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export MPLCONFIGDIR="$PWD/runs/.matplotlib"
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
mkdir -p "$MPLCONFIGDIR"
task_python="${ROBOENV_PYTHON:-/home/akhjd/ros2_ws/RoboEnv/.pixi/envs/default/bin/python}"
exec "$task_python" -B -m comp0245.pipeline "$@"
