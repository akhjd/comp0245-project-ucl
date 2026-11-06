"""Local demo page (5 Nov): pick one of our validation rooms with moving obstacles and launch the PyBullet GUI with our
learned robot (run_scene.py: room base + room arm, arm folded while driving, unfolds before reaching) or with the
window expert (the teacher). One run at a time. Serves http://127.0.0.1:8250 (local only).
Usage: python runs/demo-web/server.py"""
import json
import subprocess
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path("/home/akhjd/ros2_ws/comp0245_project")    # runs/ is a symlink to /data, so not from __file__
PY = "/home/akhjd/ros2_ws/RoboEnv/.pixi/envs/default/bin/python"
HERE = ROOT/"runs/demo-web"
FOLD = [3.14159, 2.618, 3.14159, 0.524, 0.0, 1.309, 0.0]
RESULTS = ROOT/"runs/arm-eval-rooms-full-val-dagger2/results.json"     # our full robot on these rooms, 100 s

ROOMS = {s["id"]: s for f in ("scenarios_rooms_train.json", "scenarios_rooms_val.json")
         for s in json.loads((ROOT/f).read_text()) if s["split"] == "validation"}
OURS = {e["id"]: e for e in json.loads(RESULTS.read_text())["episodes"]}
job = {"proc": None, "who": None, "room": None, "out": None, "started": None}


def movers(s):
    return sum(any(v != 0 for v in o.get("vel", [0, 0])) for o in s["obstacles"])


def launch(room, who):
    s = {**ROOMS[room], "arm_q0": FOLD}
    stamp = time.strftime("%H%M%S")
    out = HERE/"runs"/f"{who}-{room}-{stamp}"
    out.parent.mkdir(parents=True, exist_ok=True)
    scen = HERE/"runs"/f"{who}-{room}-{stamp}.json"
    if who == "ours":
        scen.write_text(json.dumps([s]))
        cmd = [PY, "-B", str(ROOT/"run_scene.py"), "--scenarios", str(scen), "--out", str(out), "--gui"]
    else:   # the teacher: window expert, arm folded, unfolds before reaching
        scen.write_text(json.dumps([{**s, "expert": "window", "expert_unfold": True}]))
        cmd = [PY, "-B", str(ROOT/"collect_obstacles.py"), "--scenarios", str(scen), "--out", str(out), "--gui",
               "--lidar-rays", "72", "--duration", "100"]
    job.update(proc=subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL),
               who=who, room=room, out=out, started=time.time())


def state():
    p = job["proc"]
    running = p is not None and p.poll() is None
    result = None
    if job["out"] is not None and (job["out"]/job["room"]/"summary.json").exists():
        r = json.loads((job["out"]/job["room"]/"summary.json").read_text())
        result = {k: r.get(k) for k in ("termination", "success", "contact_steps", "simulation_time_s", "minimum_error_m")}
    return {"running": running, "who": job["who"], "room": job["room"], "result": result,
            "seconds": round(time.time() - job["started"]) if job["started"] else None}


PAGE = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Moving Obstacles Demo</title><style>
:root{--bg:#fafaf9;--card:#fff;--ink:#1c1917;--mute:#78716c;--line:#e7e5e4;--ok:#15803d;--bad:#b91c1c;--acc:#2563eb}
@media (prefers-color-scheme:dark){:root{--bg:#161514;--card:#1f1e1d;--ink:#f5f5f4;--mute:#a8a29e;--line:#3a3836;--ok:#4ade80;--bad:#f87171;--acc:#60a5fa}}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,sans-serif}
main{max-width:980px;margin:0 auto;padding:16px}h1{font-size:22px;margin:4px 0}.mute{color:var(--mute)}
#state{position:sticky;top:0;background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 14px;margin:12px 0;z-index:2}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:10px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 12px}
.card b{font-size:14px}.tag{font-size:12px;padding:1px 7px;border-radius:9px;border:1px solid var(--line);margin-left:4px}
.ok{color:var(--ok)}.bad{color:var(--bad)}button{font:inherit;padding:5px 10px;margin:6px 6px 0 0;border-radius:7px;
border:1px solid var(--line);background:var(--bg);color:var(--ink);cursor:pointer}button.main{border-color:var(--acc);color:var(--acc)}
button:disabled{opacity:.45;cursor:default}h2{font-size:16px;margin:18px 0 6px}</style></head><body><main>
<h1>Moving obstacles: learned robot vs expert</h1>
<div class="mute">Our validation rooms with moving cylinders (never trained on). <b>Our robot</b>: learned base drives with the
arm folded, parks, unfolds, learned arm touches the red target. <b>Expert</b>: the window teacher it learned from.
Each button opens a PyBullet window; one run at a time. Our robot's 100 s result from the validation run is shown on each card.</div>
<div id="state">Idle.</div>
<h2>Our robot succeeded</h2><div class="grid" id="good"></div>
<h2>Our robot failed (honest examples)</h2><div class="grid" id="bad"></div>
<script>
const rooms = __ROOMS__;
function card(r){ return `<div class="card"><b>${r.id.replace('validation','')}</b><span class="tag">${r.movers} mover${r.movers>1?'s':''}</span>
 <span class="tag">${r.obstacles} obstacles</span><div class="${r.success?'ok':'bad'}" style="font-size:13px">${r.note}</div>
 <button class="main" onclick="run('${r.id}','ours')">Our robot</button><button onclick="run('${r.id}','expert')">Expert</button></div>`; }
document.getElementById('good').innerHTML = rooms.filter(r=>r.success).map(card).join('');
document.getElementById('bad').innerHTML = rooms.filter(r=>!r.success).map(card).join('');
async function run(room, who){ const r = await fetch(`/run?room=${room}&who=${who}`, {method:'POST'}); const j = await r.json();
  if(j.error) alert(j.error); poll(); }
async function stop(){ await fetch('/stop', {method:'POST'}); poll(); }
async function poll(){ const j = await (await fetch('/state')).json(); const el = document.getElementById('state');
  const name = j.who==='ours' ? 'Our robot' : 'Expert';
  if(j.running){ el.innerHTML = `Running <b>${name}</b> in ${j.room.replace('validation','')} (${j.seconds} s) - watch the PyBullet window. <button onclick="stop()">Stop run</button>`; }
  else if(j.room){ const r = j.result; el.innerHTML = r ? `${name} in ${j.room.replace('validation','')}: <b class="${r.success?'ok':'bad'}">${r.success?'success':r.termination}</b>, `
     + `${r.simulation_time_s.toFixed(1)} s simulated, closest hand ${(100*r.minimum_error_m).toFixed(1)} cm, contact steps ${r.contact_steps}` : `${name} run ended without a result (window closed early?).`; }
  document.querySelectorAll('.card button').forEach(b=>b.disabled=j.running); }
setInterval(poll, 1500); poll();
</script></main></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def send(self, body, kind="application/json", code=200):
        data = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/state":
            return self.send(json.dumps(state()))
        if path != "/":
            return self.send("{}", code=404)
        rooms = []
        for rid, s in ROOMS.items():
            e = OURS.get(rid)
            if e is None or movers(s) == 0:
                continue
            note = (f"validation run: success, reached in {e['reach_time_s']:.1f} s after unfolding" if e["success"] else
                    "validation run: hit a mover" if e["contact_steps"] else
                    "validation run: did not park" if e["parked_at_s"] is None else "validation run: arm missed")
            rooms.append({"id": rid, "movers": movers(s), "obstacles": len(s["obstacles"]), "success": bool(e["success"]), "note": note})
        rooms.sort(key=lambda r: (-r["movers"], r["id"]))
        self.send(PAGE.replace("__ROOMS__", json.dumps(rooms)), "text/html; charset=utf-8")

    def do_POST(self):
        url = urlparse(self.path)
        q = parse_qs(url.query)
        if url.path == "/stop":
            if job["proc"] is not None and job["proc"].poll() is None:
                job["proc"].terminate()
            return self.send(json.dumps(state()))
        if url.path == "/run":
            room, who = q.get("room", [""])[0], q.get("who", [""])[0]
            if room not in ROOMS or who not in ("ours", "expert"):
                return self.send(json.dumps({"error": "unknown room or driver"}), code=400)
            if job["proc"] is not None and job["proc"].poll() is None:
                return self.send(json.dumps({"error": "A run is still going: stop it first."}))
            launch(room, who)
            return self.send(json.dumps(state()))
        self.send("{}", code=404)


if __name__ == "__main__":
    print("Demo page: http://127.0.0.1:8250", flush=True)
    ThreadingHTTPServer(("127.0.0.1", 8250), Handler).serve_forever()
