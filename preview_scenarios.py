"""Top-down picture of scenarios (no simulator): start, parking spot, target, static obstacles (grey) and the sweep of
each moving obstacle (blue: start disc, line = its whole back-and-forth range, arrow = first direction).

Example:
    python preview_scenarios.py scenarios_obstacles_moving_mixed.json --out runs/preview-mixed.png
    python preview_scenarios.py scenarios_obstacles_moving_only.json --only trainmv_000 trainmv_001 --out runs/p.png
"""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def draw(ax, s):
    ax.set_aspect("equal"); ax.grid(alpha=.3); ax.set_title(s["id"], fontsize=9)
    if "room" in s:
        x0, y0, x1, y1 = s["room"]
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="tab:brown", lw=3))
        ax.set_xlim(x0 - .3, x1 + .3); ax.set_ylim(y0 - .3, y1 + .3)
    x, y, yaw = s["start"]
    ax.plot(x, y, "go"); ax.arrow(x, y, .4*np.cos(yaw), .4*np.sin(yaw), color="g", head_width=.08)
    ax.plot(*s["park"], "k^"); ax.plot(*s["target"][:2], "r*", ms=12)
    ax.plot([x, s["park"][0]], [y, s["park"][1]], "g:", lw=1)
    for o in s["obstacles"]:
        c = np.array(o["xy"])
        if "vel" in o:
            v = np.array(o["vel"]); d = v/np.linalg.norm(v)
            lo, hi = o.get("sweep", [-o["span"], o["span"]])
            ends = np.array([c + lo*d, c + hi*d])
            ax.plot(*ends.T, "-", color="tab:blue", alpha=.4, lw=3)
            ax.add_patch(plt.Circle(c, o["r"], color="tab:blue", alpha=.7))
            ax.arrow(*c, *(d*.6), color="tab:blue", head_width=.08)
        else:
            ax.add_patch(plt.Circle(c, o["r"], color="gray"))

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("scenarios"); parser.add_argument("--out", required=True)
parser.add_argument("--only", nargs="*"); parser.add_argument("--max", type=int, default=6)
args = parser.parse_args()
sc = json.loads(Path(args.scenarios).read_text())
sc = [s for s in sc if not args.only or s["id"] in args.only][:args.max]
cols = min(3, len(sc)); rows = -(-len(sc)//cols)
fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows), squeeze=False)
for ax, s in zip(axes.flat, sc):
    draw(ax, s)
for ax in list(axes.flat)[len(sc):]:
    ax.axis("off")
fig.tight_layout(); fig.savefig(args.out, dpi=90)
print(f"wrote {args.out}")
