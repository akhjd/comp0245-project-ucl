"""Build the row caches (train_base_torch.strided_rows, 50 ms) of the finished teacher recordings while the rest are
still recording, so DAgger's round 0 does not wait for them. Low priority, few workers. argv: folders..."""
import json, sys
from pathlib import Path
sys.path.insert(0, "/home/akhjd/ros2_ws/comp0245_project")
import train_base_torch as tb

if __name__ == "__main__":
    tb.set_stride(50)
    tb.CACHE_WORKERS = 3
    for f in map(Path, sys.argv[1:]):
        m = json.loads((f/"manifest.json").read_text())
        eps = [e for e in m["episodes"] if e["split"] == "train" and e["success"]]
        tb.build_caches(f, eps, max(1, tb.STRIDE // m.get("saved_every", 1)))
        print(f, len(eps), "episodes cached", flush=True)
