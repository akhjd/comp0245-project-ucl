"""Training rooms for the 5-6 Nov combined overnight run: our room types and Julia's single-target cases, the same
number of each, all training split, ids that say the room type (train_base_torch.group_of = the id up to "_"):
    ovrmv_    our rooms with 2-3 movers           (make_obstacle_scenarios.py settings of scenarios_rooms_train.json)
    ovrmx_    our rooms with 1-2 movers + 1-3 still cylinders
    ovc2_     Julia's case2: 10 x 10 m room, still and moving cylinders   (her make_dataset.make, run from her clone)
    ovc3_     Julia's case3: 8 x 8 m floor plan, walls with doorways, furniture, no cylinders
    ovc4_     Julia's case4: floor plan + cylinders (at least one)
    ovd3_     doorway rooms: floor plan, no cylinders, start and target ALWAYS in different rooms (her cross_rooms = 1)
    ovd4_     doorway rooms with cylinders
    ovx_      our crossing-mover scenes: an ovrmv / ovrmx room with one extra mover timed to cut across the robot's way
              (runs/crossing/make_crossing.crossing, robot assumed at 0.12-0.35 m/s); for DAgger's hard pool
Every scenario gets the teacher settings: expert window_fast at TOP m/s, arm folded, expert_unfold.
Usage: make_rooms.py OUT_FOLDER PER_TYPE TOP_SPEED [SEED]"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path
import numpy as np

ROOT = Path("/home/akhjd/ros2_ws/comp0245_project")
JULIA = Path("/home/akhjd/ros2_ws/comp0245_julia")
PY = "/home/akhjd/ros2_ws/RoboEnv/.pixi/envs/default/bin/python"
FOLD = [3.14159, 2.618, 3.14159, 0.524, 0.0, 1.309, 0.0]


def ours(kind, n, seed):
    """Our room generator, same settings as scenarios_rooms_train.json; ids re-made as ov<kind>_NNNN."""
    counts = ["--moving-count", "2", "3"] if kind == "rmv" else ["--moving-count", "1", "2", "--static-count", "1", "3"]
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)/"rooms.json"
        subprocess.run([PY, "-B", "make_obstacle_scenarios.py", "--out", str(out), "--train", str(n), "--validation", "0",
                        "--test", "0", "--seed", str(seed), "--room-half", "3.5", "7", *counts], cwd=ROOT, check=True,
                       stdout=subprocess.DEVNULL)
        rooms = json.loads(out.read_text())
    return [{**s, "id": f"ov{kind}_{k:04d}", "split": "train"} for k, s in enumerate(rooms)]


def doors(cylinders, n, seed):
    """Julia's floor-plan generator (make_floorplan_scenarios.sample, from her clone) with start and target always in
    different rooms, so every room needs a doorway."""
    statics, movers = ("[0, 3]", 0.7) if cylinders else ("[0, 0]", 0.0)
    code = (f"import sys, json, types, numpy as np; sys.path.insert(0, {str(JULIA)!r}); import make_floorplan_scenarios as fp; "
            f"rng = np.random.default_rng([{seed}, {7 if cylinders else 6}]); "
            f"a = types.SimpleNamespace(cross_rooms=1.0, mover_share={movers}, dots=1, height=[0.35, 0.8], "
            f"static_count={statics}, radius=[0.15, 0.3]); out = []\n"
            f"while len(out) < {n}:\n    s = fp.sample(rng, 'random', a)\n"
            f"    if {cylinders} and not s['obstacles']: continue\n"
            f"    out.append({{**s, 'seed': int(rng.integers(1_000_000))}})\n"
            f"print(json.dumps(out))")
    text = subprocess.run([PY, "-B", "-c", code], cwd=JULIA, check=True, capture_output=True, text=True).stdout
    rooms = json.loads(text.strip().splitlines()[-1])
    return [{**s, "id": f"ovd{4 if cylinders else 3}_{k:04d}", "split": "train"} for k, s in enumerate(rooms)]


def crossings(rooms, seed):
    """One crossing-mover variant per room where it fits (make_crossing.crossing), ids ovx_NNNN."""
    sys.path.insert(0, str(ROOT/"runs/crossing"))
    from make_crossing import crossing
    rng = np.random.default_rng(seed)
    out = []
    for s in rooms:
        m = crossing(s, rng, robot_speed=(0.12, 0.35))
        if m is not None:
            out.append({**s, "id": f"ovx_{len(out):04d}", "obstacles": s["obstacles"] + [m]})
    return out


def hers(case, n, seed):
    """Julia's generator (her make_dataset.make, from her clone); a new seed, so none of her own scenarios repeat."""
    code = (f"import sys, json, numpy as np; sys.path.insert(0, {str(JULIA)!r}); import make_dataset as md; "
            f"rng = np.random.default_rng([{seed}, {int(case[-1])}]); "
            f"print(json.dumps([{{**md.make({case!r}, rng), 'case': {case!r}, 'seed': int(rng.integers(1_000_000))}} "
            f"for _ in range({n})]))")
    text = subprocess.run([PY, "-B", "-c", code], cwd=JULIA, check=True, capture_output=True, text=True).stdout
    rooms = json.loads(text.strip().splitlines()[-1])
    return [{**s, "id": f"ov{case.replace('case', 'c')}_{k:04d}", "split": "train"} for k, s in enumerate(rooms)]


if __name__ == "__main__":
    out, per, top = Path(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 20261106
    out.mkdir(parents=True, exist_ok=True)
    sets = {"ovrmv": ours("rmv", per, seed), "ovrmx": ours("rmx", per, seed + 1),
            "ovc2": hers("case2", per, seed), "ovc3": hers("case3", per, seed), "ovc4": hers("case4", per, seed),
            "ovd3": doors(False, per, seed), "ovd4": doors(True, per, seed)}
    sets["ovx"] = crossings(sets["ovrmv"] + sets["ovrmx"], seed + 2)
    teacher = {"expert": "window_fast", "top_speed": top, "expert_unfold": True, "arm_q0": FOLD}
    for name, rooms in sets.items():
        rooms = [{**s, **teacher} for s in rooms]
        (out/f"{name}.json").write_text(json.dumps(rooms, indent=1))
        print(name, len(rooms), "rooms,", sum(any(np.any(o.get("vel", [0, 0])) for o in s["obstacles"]) for s in rooms),
              "with a mover", flush=True)
