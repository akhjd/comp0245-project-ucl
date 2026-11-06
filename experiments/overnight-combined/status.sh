#!/bin/bash
# Status of the combined overnight run (and its smoke test) with progress bars. Read-only.
# Use: watch -n 20 runs/overnight-combined/status.sh
exec /home/akhjd/ros2_ws/RoboEnv/.pixi/envs/default/bin/python -B /home/akhjd/ros2_ws/comp0245_project/runs/overnight-combined/status.py
