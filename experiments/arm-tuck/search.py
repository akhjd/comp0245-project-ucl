"""Search for a folded arm pose (4 Nov): every arm part inside the base footprint (x +-0.342, y +-0.313), at least
3 cm from the robot's own base and wheels, as low as possible (a lower arm moves the centre of mass less). Joints 2, 4,
6 (the pitch joints) searched on a grid, the others 0 or pi; within the URDF limits."""
import itertools
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path("/home/akhjd/ros2_ws/comp0245_project/runs/arm-tuck")))
from probe import robot, arm_joints, report

if __name__ == "__main__":
    r = robot()
    c = r.client
    joints = arm_joints(r)
    limits = [c.getJointInfo(r.rid, j)[8:10] for j in joints]
    print("limits", [tuple(round(x, 2) for x in l) for l in limits], flush=True)
    arm_links = [j for j in range(c.getNumJoints(r.rid)) if "j2s7s300" in c.getJointInfo(r.rid, j)[12].decode()]
    body_links = [-1] + [j for j in range(c.getNumJoints(r.rid)) if j not in arm_links]
    skip = {j for j in arm_links if c.getJointInfo(r.rid, j)[12].decode() in ("robot_j2s7s300_link_base", "robot_j2s7s300_link_1")}

    def clearance():
        d = np.inf
        for a in arm_links:
            if a in skip:
                continue
            for b in body_links:
                pts = c.getClosestPoints(r.rid, r.rid, 0.2, linkIndexA=a, linkIndexB=b)
                if pts:
                    d = min(d, min(p[8] for p in pts))
        return d

    def ok(q):
        return all(lo > hi or lo - 1e-9 <= a <= hi + 1e-9 for a, (lo, hi) in zip(q, limits))   # lo > hi: continuous joint

    grid = np.radians(np.arange(-180, 181, 15))
    best = []
    for q1, q3, q5, q7 in itertools.product((0.0, np.pi), (0.0, np.pi), (0.0,), (0.0,)):
        for q2, q4, q6 in itertools.product(grid, grid, grid):
            q = [q1, q2, q3, q4, q5, q6, q7]
            if not ok(q):
                continue
            rep = report(r, q)
            if rep["outside_footprint_any_m"] > 0:
                continue
            gap = clearance()
            if gap < 0.03:
                continue
            best.append({k: v for k, v in rep.items() if k != "links"} | {"clearance_m": round(float(gap), 3)})
    best.sort(key=lambda b: b["top_z"])
    print(len(best), "poses inside the footprint with >= 3 cm clearance")
    for b in best[:10]:
        print(b)
    (Path(__file__).parent/"search.json").write_text(json.dumps(best[:50], indent=1))
    r.close()
