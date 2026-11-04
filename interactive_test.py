"""Interactive test of the wheels MLP: build a scenario by clicking, then watch the MLP (or the expert) drive it in PyBullet.

    python interactive_test.py
    python interactive_test.py --model runs/dagger-pre/r3/best.pt

Left: the 8 x 8 m room seen from above. Right: controls.
  Floor plan    "Random plan" draws a new random floor plan (walls with 1.4 m doorways, furniture); "Open room" removes it;
                "Random case 3/4/5" draws a whole dataset-style scenario (plan, robot, dots, cylinders) to edit.
                "Load id" loads a saved dataset scenario, such as case4_0169, typed in the box.
  Click mode    Robot: click to place the robot.  Heading: click where the robot should face.
                Dot: click to add a dot (up to 3; the visiting order is the expert's choice, as in the dataset).
                Static cyl: click to add a still cylinder.  Moving cyl: press at its start, drag to where it turns round.
                Erase: click next to a dot or cylinder to remove it.
  Values        cylinder radius (m), mover speed (m/s), dot height (m).
  Run           "Run MLP" opens the PyBullet GUI with the MLP driving (drive_mlp.py); "Run expert" with the RRT expert
                (collect_obstacles.py --rrt). The result appears in the status line when the run ends.
Every run's scenario is saved in runs/interactive/<name>.json and the run in runs/interactive/mlp/ or .../expert-<name>/.
Checks before a run: the robot and every dot's parking spot are clear of walls, furniture and cylinders, and a route exists.
"""
import argparse
import glob
import json
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrow, Rectangle
from matplotlib.widgets import Button, RadioButtons, TextBox
import make_floorplan_scenarios as fp
from obstacle_env import ROBOT_RADIUS, FloorPlanRoute, standoff

ROOT = Path(__file__).resolve().parent
ROOM = [-4.0, -4.0, 4.0, 4.0]
OUT = ROOT/"runs/interactive"
MODES = ("Robot", "Heading", "Dot", "Static cyl", "Moving cyl", "Erase")


class Editor:
    def __init__(self, model):
        self.model = model
        self.boxes, self.start, self.dots, self.statics, self.movers = [], [-3.0, -3.0, 0.0], [], [], []
        self.rng = np.random.default_rng()
        self.press, self.job = None, None
        self.fig = plt.figure(figsize=(13, 8.2))
        self.fig.canvas.manager.set_window_title("MLP interactive test")
        self.ax = self.fig.add_axes([0.03, 0.08, 0.6, 0.88])
        self.status = self.fig.text(0.03, 0.025, "", fontsize=10)
        x, w = 0.67, 0.14
        self.radio = RadioButtons(self.fig.add_axes([x, 0.66, w, 0.28]), MODES)
        self.fig.text(x, 0.95, "Click mode", fontsize=11, weight="bold")
        buttons = [("Random plan", self.random_plan), ("Open room", self.open_room), ("Random case 3", lambda: self.random_case("case3")),
                   ("Random case 4", lambda: self.random_case("case4")), ("Random case 5", lambda: self.random_case("case5")),
                   ("Clear cylinders", self.clear_cylinders), ("Clear dots", self.clear_dots)]
        self.widgets = []
        for i, (label, action) in enumerate(buttons):
            b = Button(self.fig.add_axes([x + 0.16, 0.89 - 0.055*i, 0.15, 0.045]), label)
            b.on_clicked(lambda _, a=action: (a(), self.draw()))
            self.widgets.append(b)
        self.fig.text(x + 0.16, 0.95, "Floor plan and scene", fontsize=11, weight="bold")
        self.load_box = TextBox(self.fig.add_axes([x + 0.05, 0.49, 0.1, 0.045]), "Load id ", initial="case4_0169")
        b = Button(self.fig.add_axes([x + 0.16, 0.49, 0.15, 0.045]), "Load id")
        b.on_clicked(lambda _: (self.load(self.load_box.text.strip()), self.draw()))
        self.widgets.append(b)
        self.radius = TextBox(self.fig.add_axes([x + 0.12, 0.41, 0.06, 0.045]), "cylinder radius m ", initial="0.2")
        self.speed = TextBox(self.fig.add_axes([x + 0.12, 0.355, 0.06, 0.045]), "mover speed m/s ", initial="0.2")
        self.height = TextBox(self.fig.add_axes([x + 0.12, 0.30, 0.06, 0.045]), "dot height m ", initial="0.5")
        self.name = TextBox(self.fig.add_axes([x + 0.12, 0.235, 0.17, 0.045]), "test name ", initial="my_test")
        for label, action, y in (("Run MLP", lambda: self.run("mlp"), 0.15), ("Run expert", lambda: self.run("expert"), 0.09)):
            b = Button(self.fig.add_axes([x + 0.05, y, 0.26, 0.05]), label, color="#cfe8cf" if label == "Run MLP" else "#e8dccf")
            b.on_clicked(lambda _, a=action: a())
            self.widgets.append(b)
        self.fig.text(x + 0.05, 0.05, f"model: {model}", fontsize=8)
        self.fig.canvas.mpl_connect("button_press_event", self.on_press)
        self.fig.canvas.mpl_connect("button_release_event", self.on_release)
        self.timer = self.fig.canvas.new_timer(interval=1000)
        self.timer.add_callback(self.poll)
        self.timer.start()
        self.random_case("case3")
        self.draw()

    # ---------- scene ----------
    def number(self, box, default):
        try:
            return float(box.text)
        except ValueError:
            return default

    def random_plan(self):
        self.boxes = fp.random_layout(self.rng)["boxes"]
        self.say("New random floor plan. Place the robot, dots and cylinders, then Run.")

    def open_room(self):
        self.boxes = []
        self.say("Open room (outer walls only).")

    def clear_cylinders(self):
        self.statics, self.movers = [], []

    def clear_dots(self):
        self.dots = []

    def use(self, s):
        self.boxes = s.get("boxes", [])
        self.start = list(s["start"])
        self.dots = [list(t) for t in s.get("targets", [s["target"]])]
        self.statics = [o for o in s["obstacles"] if not np.any(np.asarray(o.get("vel", [0, 0])) != 0)]
        self.movers = [o for o in s["obstacles"] if np.any(np.asarray(o.get("vel", [0, 0])) != 0)]

    def random_case(self, case):
        import make_dataset
        self.use(make_dataset.make(case, self.rng))
        self.say(f"Random {case} scenario drawn like the dataset (not one the MLP was trained on). Edit it or Run.")

    def load(self, sid):
        found = glob.glob(str(ROOT/f"runs/dataset/*/part*/{sid}*/episode.json"))
        if not found:
            self.say(f"No dataset scenario {sid}")
            return
        s = json.loads(Path(found[0]).read_text())["scenario"]
        self.use(s)
        self.say(f"Loaded {sid} ({s['split']} split).")

    # ---------- clicks ----------
    def on_press(self, e):
        if e.inaxes is not self.ax or e.xdata is None:
            return
        p, mode = np.array([e.xdata, e.ydata]), self.radio.value_selected
        r = self.number(self.radius, 0.2)
        if mode == "Robot":
            self.start[:2] = [round(float(v), 3) for v in p]
        elif mode == "Heading":
            self.start[2] = round(float(np.arctan2(*(p - self.start[:2])[::-1])), 3)
        elif mode == "Dot":
            if len(self.dots) >= 3:
                self.say("At most 3 dots (as in case 5). Erase one first.")
            else:
                self.dots.append([round(float(p[0]), 3), round(float(p[1]), 3), self.number(self.height, 0.5)])
        elif mode == "Static cyl":
            self.statics.append({"xy": [round(float(v), 3) for v in p], "r": r})
        elif mode == "Moving cyl":
            self.press = p
            return
        elif mode == "Erase":
            items = [("dot", i, np.array(d[:2])) for i, d in enumerate(self.dots)] + \
                    [("static", i, np.array(o["xy"])) for i, o in enumerate(self.statics)] + \
                    [("mover", i, np.array(o["xy"])) for i, o in enumerate(self.movers)]
            if items:
                kind, i, q = min(items, key=lambda it: np.linalg.norm(it[2] - p))
                if np.linalg.norm(q - p) < 0.5:
                    {"dot": self.dots, "static": self.statics, "mover": self.movers}[kind].pop(i)
        self.draw()

    def on_release(self, e):
        if self.press is None:
            return
        a, self.press = self.press, None
        if e.xdata is None:
            return
        b = np.array([e.xdata, e.ydata])
        length = float(np.linalg.norm(b - a))
        if length < 0.3:
            self.say("Moving cylinder: press at its start and drag (at least 0.3 m) to where it turns round.")
            return
        d = (b - a)/length
        self.movers.append({"xy": [round(float(v), 3) for v in a], "r": self.number(self.radius, 0.2),
                            "vel": [round(float(v), 3) for v in d*self.number(self.speed, 0.2)], "sweep": [0.0, round(length, 3)]})
        self.draw()

    # ---------- scenario ----------
    def scenario(self):
        """The scene as a scenario dict like the dataset's, or (None, reason) if it cannot run."""
        if not self.dots:
            return None, "Add at least one dot."
        lo = np.array([b["min"] for b in self.boxes], float).reshape(-1, 2)
        hi = np.array([b["max"] for b in self.boxes], float).reshape(-1, 2)
        room_lo, room_hi = np.array(ROOM[:2]), np.array(ROOM[2:])
        start = np.array(self.start[:2])
        cylinders = self.statics + self.movers
        if np.any(start < room_lo + ROBOT_RADIUS + 0.05) or np.any(start > room_hi - ROBOT_RADIUS - 0.05) or \
                (len(lo) and not fp.clear_of(start, lo, hi, ROBOT_RADIUS + 0.05)) or \
                any(np.linalg.norm(start - o["xy"]) < o["r"] + ROBOT_RADIUS + 0.1 for o in cylinders):
            return None, "The robot overlaps a wall, furniture or a cylinder: move it."
        route = FloorPlanRoute(ROOM, lo, hi, [np.array(o["xy"]) for o in self.statics], [o["r"] for o in self.statics])
        parks = []
        for k, (tx, ty, tz) in enumerate(self.dots):
            t = np.array([tx, ty])
            if np.any(t < room_lo + 0.05) or np.any(t > room_hi - 0.05):
                return None, f"Dot {k + 1} is outside the room."
            first = np.arctan2(*(t - start)[::-1])                      # prefer parking on the robot's side of the dot
            for bearing in first + np.radians(np.r_[0, np.repeat(np.arange(15, 181, 15), 2)*np.tile([1, -1], 12)]):
                q = t - standoff(tz)*np.array([np.cos(bearing), np.sin(bearing)])
                if np.any(q < room_lo + ROBOT_RADIUS + 0.2) or np.any(q > room_hi - ROBOT_RADIUS - 0.2):
                    continue
                if len(lo) and not (fp.clear_of(q, lo, hi, ROBOT_RADIUS + 0.2) and
                                    all(fp.clear_of(q + s*(t - q), lo, hi, 0.2) for s in np.linspace(0, 1, 8))):
                    continue
                if any(np.linalg.norm(q - o["xy"]) < o["r"] + ROBOT_RADIUS + 0.2 for o in self.statics):
                    continue
                if route.path(start, q) is None:
                    continue
                parks.append([round(float(v), 4) for v in q])
                break
            else:
                return None, f"Dot {k + 1}: no free parking spot next to it with a route from the robot. Move it."
        name = "".join(c for c in self.name.text.strip() if c.isalnum() or c in "-_") or "my_test"
        s = {"layout": "interactive", "room": ROOM, "boxes": self.boxes, "fixed_park": True, "start": list(self.start),
             "target": list(self.dots[0]), "park": parks[0], "obstacles": self.statics + self.movers, "id": name,
             "split": "interactive", "case": "interactive", "dot_order": "best", "expert": "rrt", "seed": 0}
        if len(self.dots) > 1:
            s.update(targets=[list(d) for d in self.dots], parks=parks)
        return s, None

    def run(self, who):
        if self.job is not None and self.job[0].poll() is None:
            self.say("A run is still open: close its PyBullet window (or wait for it to end) first.")
            return
        s, problem = self.scenario()
        if problem:
            self.say(problem)
            return
        OUT.mkdir(parents=True, exist_ok=True)
        path = OUT/f"{s['id']}.json"
        path.write_text(json.dumps([s], indent=1))
        if who == "mlp":
            out = OUT/"mlp"
            cmd = [sys.executable, str(ROOT/"drive_mlp.py"), "--scenario-file", str(path), "--model", self.model, "--gui", "--out", str(out)]
        else:
            out = OUT/f"expert-{s['id']}-{time.strftime('%H%M%S')}"
            cmd = [sys.executable, str(ROOT/"collect_obstacles.py"), "--scenarios", str(path), "--out", str(out), "--rrt", "--gui",
                   "--duration", "120"]
        self.job = (subprocess.Popen(cmd, cwd=ROOT), out/s["id"]/"summary.json", who)
        self.say(f"Running the {'MLP' if who == 'mlp' else 'expert'} on {s['id']} in the PyBullet window... (saved as {path.name})")

    def poll(self):
        if self.job is None or self.job[0].poll() is None:
            return
        proc, summary, who = self.job
        self.job = None
        if summary.exists():
            r = json.loads(summary.read_text())
            self.say(f"{'MLP' if who == 'mlp' else 'Expert'}: {r['termination']}, clean {r['success']}, dots {r['dots_picked']}/{r['dots_total']}, "
                     f"contact steps {r['contact_steps']}, {r['simulation_time_s']:.1f} s simulated")
        else:
            self.say(f"The {who} run ended without a result (window closed early, or an error: see the terminal).")

    # ---------- drawing ----------
    def say(self, text):
        self.status.set_text(text)
        self.fig.canvas.draw_idle()

    def draw(self):
        ax = self.ax
        ax.clear()
        ax.set_xlim(-4.3, 4.3)
        ax.set_ylim(-4.3, 4.3)
        ax.set_aspect("equal")
        ax.grid(alpha=0.25)
        ax.add_patch(Rectangle((-4, -4), 8, 8, fill=False, lw=3, color="#333"))
        for b in self.boxes:
            (x0, y0), (x1, y1) = b["min"], b["max"]
            furniture = b.get("kind") == "furniture"
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, color="#b08850" if furniture else "#333", alpha=0.8 if furniture else 1))
        for o in self.statics:
            ax.add_patch(Circle(o["xy"], o["r"], color="#777"))
        for o in self.movers:
            a, v = np.array(o["xy"]), np.array(o["vel"])
            d = v/np.linalg.norm(v)
            lo, hi = o.get("sweep", [0, 1])
            ax.plot(*np.array([a + lo*d, a + min(hi, 9)*d]).T, "--", color="#3a6fd8", lw=1.5)
            ax.add_patch(Circle(a, o["r"], color="#3a6fd8"))
            ax.annotate(f"{np.linalg.norm(v):.2f} m/s", a + 0.3, color="#3a6fd8", fontsize=8)
        x, y, yaw = self.start
        ax.add_patch(Circle((x, y), ROBOT_RADIUS, fill=False, lw=2, color="#1a8a3a"))
        ax.add_patch(FancyArrow(x, y, 0.5*np.cos(yaw), 0.5*np.sin(yaw), width=0.04, color="#1a8a3a"))
        for k, d in enumerate(self.dots):
            ax.plot(d[0], d[1], "o", color="#d62728", ms=11)
            ax.annotate(f"{k + 1} (z {d[2]:.2f})", (d[0] + 0.15, d[1] + 0.15), color="#d62728", fontsize=9)
        ax.set_title("green: robot   red: dots   grey: still cylinders   blue: movers (dashed: their path)   "
                     "black: walls   brown: furniture", fontsize=9)
        self.fig.canvas.draw_idle()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="runs/dagger-pre/r3/best.pt")
    args = parser.parse_args()
    if matplotlib.get_backend().lower() == "agg":
        raise SystemExit("No display for the window (matplotlib backend is Agg)")
    Editor(args.model)
    plt.show()


if __name__ == "__main__":
    main()
