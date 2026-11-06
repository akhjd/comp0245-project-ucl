"""Fresh validation rooms (6 Nov) to confirm a round picked on val_all.json: same generators and the same mix as
val_all (ours 30 + 30 like scenarios_rooms_val.json; Julia's case2 20, case3 15, case4 25 like val_case234_fold.json),
new seeds, never used for training or for choosing. Ids: freshrmv_ / freshrmx_ (ours), freshc2_ / freshc3_ / freshc4_.
Usage: make_fresh_val.py OUT.json [SEED]"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "/home/akhjd/ros2_ws/comp0245_project/runs/overnight-combined")
import make_rooms as mr

if __name__ == "__main__":
    out = Path(sys.argv[1])
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 20261199
    teacher = {"expert": "window_fast", "top_speed": 0.8, "expert_unfold": True, "arm_q0": mr.FOLD}
    sets = [("freshrmv", mr.ours("rmv", 30, seed)), ("freshrmx", mr.ours("rmx", 30, seed + 1)),
            ("freshc2", mr.hers("case2", 20, seed)), ("freshc3", mr.hers("case3", 15, seed)),
            ("freshc4", mr.hers("case4", 25, seed))]
    rooms = [{**s, **teacher, "id": f"{name}_{k:03d}", "split": "validation"} for name, group in sets
             for k, s in enumerate(group)]
    out.write_text(json.dumps(rooms, indent=1))     # overwrites nothing: checked below
    print(len(rooms), "fresh validation rooms ->", out)
