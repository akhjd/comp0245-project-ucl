"""Arm pose probe (4 Nov): where do the arm's parts sit for a given joint pose, relative to the base footprint
(wheel boxes x +-0.342, y +-0.313) and the 0.6 m obstacle tops? Base at the origin facing +x."""
import contextlib
import io
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path("/home/akhjd/ros2_ws/comp0245_project")
sys.path.insert(0, str(ROOT))
from obstacle_env import ObstacleRobot
from pipeline import read_json

HX, HY, TOP = 0.342, 0.313, 0.6


def robot():
    config = read_json(ROOT/"settings.json")
    scenario = {"id": "tuck", "split": "validation", "seed": 0, "start": [0.0, 0.0, 0.0], "target": [3.0, 0.0, 0.6],
                "park": [2.0, 0.0], "obstacles": []}
    with contextlib.redirect_stdout(io.StringIO()):
        return ObstacleRobot(config, scenario)


def arm_joints(r):
    c = r.client
    names = {c.getJointInfo(r.rid, j)[1].decode(): j for j in range(c.getNumJoints(r.rid))}
    return [names[f"robot_j2s7s300_joint_{k}"] for k in range(1, 8)]


def report(r, q=None):
    c = r.client
    joints = arm_joints(r)
    if q is not None:
        for j, a in zip(joints, q):
            c.resetJointState(r.rid, j, a, 0.0)
    base_z = c.getBasePositionAndOrientation(r.rid)[0][2]
    rows = []
    for j in range(c.getNumJoints(r.rid)):
        name = c.getJointInfo(r.rid, j)[12].decode()
        if "j2s7s300" not in name:
            continue
        lo, hi = c.getAABB(r.rid, j)
        rows.append((name, np.round(lo, 3).tolist(), np.round(hi, 3).tolist()))
    lo = np.array([a[1] for a in rows]); hi = np.array([a[2] for a in rows])
    low = lo[:, 2] < TOP                        # parts that reach below the obstacle tops
    out_x = np.maximum(np.maximum(hi[:, 0] - HX, -HX - lo[:, 0]), 0)
    out_y = np.maximum(np.maximum(hi[:, 1] - HY, -HY - lo[:, 1]), 0)
    return {"q": [round(float(a), 3) for a in (q if q is not None else [c.getJointState(r.rid, j)[0] for j in joints])],
            "reach_x": round(float(hi[:, 0].max()), 3), "reach_y": round(float(np.abs(np.r_[lo[:, 1], hi[:, 1]]).max()), 3),
            "top_z": round(float(hi[:, 2].max()), 3), "lowest_arm_z": round(float(lo[:, 2].min()), 3),
            "outside_footprint_low_m": round(float(np.hypot(out_x, out_y)[low].max()) if low.any() else 0.0, 3),
            "outside_footprint_any_m": round(float(np.hypot(out_x, out_y).max()), 3), "base_z": round(base_z, 3), "links": rows}


if __name__ == "__main__":
    r = robot()
    c = r.client
    joints = arm_joints(r)
    print("limits", [tuple(round(x, 2) for x in c.getJointInfo(r.rid, j)[8:10]) for j in joints])
    start = report(r)
    print("start pose", {k: v for k, v in start.items() if k != "links"})
    for name, lo, hi in start["links"]:
        print("  ", name, lo, hi)
    r.close()
