"""Status of the 5-6 Nov combined overnight run (runs/overnight-combined/run.sh), with progress bars. Read-only.
Use: watch -n 20 runs/overnight-combined/status.sh"""
import json
import re
import subprocess
import time
from pathlib import Path

ROOT = Path("/home/akhjd/ros2_ws/comp0245_project")
TYPES = ["ovrmv", "ovrmx", "ovc2", "ovc3", "ovc4", "ovd3", "ovd4"]
NAMES = {"ovrmv": "our rooms, 2-3 movers", "ovrmx": "our rooms, movers + still", "ovc2": "Julia case 2 (open room)",
         "ovc3": "Julia case 3 (floor plan)", "ovc4": "Julia case 4 (plan + cyl.)", "ovd3": "doorway rooms",
         "ovd4": "doorway rooms + cyl.", "ovd3b": "doorway rooms, rest (6 Nov)", "val_all": "validation rooms (ours + hers)"}


def bar(done, total, width=28):
    total = max(total, 1)
    k = int(round(width*min(done, total)/total))
    return f"[{'#'*k}{'.'*(width - k)}] {done:>4}/{total:<4}"


def count(folder):
    return sum(1 for p in folder.iterdir() if p.is_dir()) if folder.is_dir() else 0


def rooms(path):
    try:
        return len(json.loads(path.read_text()))
    except (OSError, ValueError):
        return 0


def running(pattern):
    return subprocess.run(["pgrep", "-f", pattern], capture_output=True).returncode == 0


def stage_lines(o, name, rounds, per_round, title):
    out = [title]
    out.append(f"  1. rooms                         {'done' if (o/'rooms/ovx.json').exists() else 'pending'}")
    out.append("  2. teacher recordings (0.8 m/s expert, 144 rays / 10 m)")
    for g in TYPES + (["ovd3b"] if (o/"rooms/ovd3b.json").exists() else []) + ["val_all"]:
        total, d = rooms(o/"rooms"/f"{g}.json"), o/"data"/g
        state = "done" if (o/"data"/f"{g}.done").exists() else ("recording" if d.is_dir() else "waiting")
        out.append(f"     {NAMES[g]:<31} {bar(count(d) if state != 'done' else total, total)} {state}")
    b = o/"baseline-hers"
    out.append(f"  3. baseline (current model, Julia's rooms) {bar(count(b), 60 if name.endswith('combined') else 3)} "
               f"{'done' if (b/'evaluation.json').exists() else 'running' if b.is_dir() else 'waiting'}")
    d = ROOT/"runs"/f"dagger-{name}"
    log = ROOT/"runs"/f"dagger-{name}.log"
    out.append(f"  4. DAgger ({rounds} rounds, {per_round} runs each, half crossing scenes; 128-unit network)")
    text = log.read_text(errors="ignore") if log.exists() else ""
    for k in range(rounds + 1):
        parts = []
        if k == 0:
            parts.append("train on teacher data: " + ("done" if (d/"r0/model.pt").exists() else "training" if d.is_dir() else "waiting"))
        else:
            parts.append("drive " + bar(count(d/f"r{k}-data") + count(d/f"r{k}-hard-data"), per_round, 16))
            parts.append("train " + ("done" if (d/f"r{k}/model.pt").exists() else "-"))
            if k == rounds:
                parts.append("validate " + bar(count(d/f"r{k}-validation"), 120 if name.endswith("combined") else 6, 12))
        out.append(f"     round {k}: " + " | ".join(parts))
    epochs = re.findall(r"^epoch\s+(\d+).*validation loss ([\d.]+)", text, re.M)
    if epochs and running("dagger.py --name " + name):
        out.append(f"     last training line: epoch {epochs[-1][0]}, validation loss {epochs[-1][1]}")
    log_run = o/("smoke.log" if "smoke" in name else "run.log")
    res = re.findall(r"^RESULT.*", log_run.read_text(errors="ignore"), re.M) if log_run.exists() else []
    out += ["  5. result (rule: ours >= 35 parked with <= 13 mover hits, and hers >= current + 10)"] + [f"     {r}" for r in res]
    return out


if __name__ == "__main__":
    lines = [f"Combined overnight run  {time.strftime('%a %H:%M:%S')}", ""]
    lines += ["DONE TODAY",
              "  [x] fast expert (0.8 m/s, 144-ray 10 m lidar): ours 55/60 (1 mover hit), Julia's 55/60 (no contacts)",
              "  [x] room arm in weights/arm_rooms.pt; Julia's bot uses it; 'Run ours' button in her tool",
              "  [x] training: both room setups, doorway rooms, target-blocked input, dropout, per-type balancing (89 tests)", ""]
    o = ROOT/"runs/overnight-combined"
    if (o/"smoke").is_dir() and not (o/"run.log").exists():
        lines += stage_lines(o/"smoke", "overnight-smoke", 1, 8, "SMOKE TEST (a few rooms, 1 DAgger round) - must pass before the real run") + [""]
    if (o/"run.log").exists():
        started = re.search(r"START (.*)", (o/"run.log").read_text(errors="ignore"))
        lines += stage_lines(o, "overnight-combined", 3, 200, f"OVERNIGHT RUN (started {started.group(1) if started else '?'})") + [""]
    else:
        lines += ["OVERNIGHT RUN: not started yet (starts after a clean smoke test)", ""]
    mem = subprocess.run(["free", "-h"], capture_output=True, text=True).stdout.splitlines()[:2]
    load = open("/proc/loadavg").read().split()[:3]
    lines += mem + [f"load average {' '.join(load)}"]
    print("\n".join(lines))
