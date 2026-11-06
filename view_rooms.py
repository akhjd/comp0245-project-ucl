"""Build one page comparing several models on the same room scenarios: an outcome grid (scenario x model) and
top-down replays of every model on the chosen scenario, side by side on one clock, with walls and moving obstacles.

Each model is NAME=FOLDER, a folder of evaluated room runs (evaluation.json + manifest.json, as dagger.py's
rK-validation) or of expert runs (manifest.json only). With no models given, the 3 Nov room comparison is used.

Example:
    python view_rooms.py --out runs/rooms-compare.html "Expert=runs/rooms-expert-val" \
        "Filter r3=runs/dagger-rooms-conv/r3-validation"
Then open the printed page in a browser.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from eval_obstacles import reach_limit, REACH_MIN_M, FACING_LIMIT
from rescore import rescore_episode

ROOT = Path(__file__).resolve().parent
STEP_S = 0.2   # replay sample period; the page interpolates between samples
HIT = {"none": 0, "mover": 1, "static": 2, "wall": 3}


def default_models():
    models = [("Expert (visible RRT)", "runs/rooms-expert-val"),
              ("Final base (open floor)", "runs/scans-rooms/base-final-rerun/validation"),
              ("3 scans, DAgger r3", "runs/dagger-rooms-3scans/r3-validation"),
              ("Filter, DAgger r3", "runs/dagger-rooms-conv/r3-validation"),
              ("Threat, DAgger r3", "runs/dagger-rooms-threat/r3-validation"),
              ("Filter + threat, DAgger r3", "runs/dagger-rooms-conv-threat/r3-validation")]
    for label, run in (("Mixture", "dagger-rooms-mix"), ("Steady expert", "dagger-rooms-steady")):
        done = sorted((ROOT/"runs"/run).glob("r*-validation/evaluation.json"))   # latest finished round
        if done:
            models.append((f"{label}, DAgger {done[-1].parent.name.split('-')[0]}", str(done[-1].parent.relative_to(ROOT))))
    return models


def load_run(folder, ep, moving):
    """One episode thinned to STEP_S: base pose in cm / centiradians, obstacle centres in cm, and per sample the worst
    contact in that stretch (HIT codes), so a brief touch between samples is not lost."""
    with np.load(folder/ep/"trajectory.npz", allow_pickle=False) as z:
        t, base, obs = z["time_s"], z["base_m"], z["observations"]
        xy = z["obstacle_xy"] if "obstacle_xy" in z.files else None
        hit_obst = z["contact_obstacle"] if "contact_obstacle" in z.files else np.zeros((len(t), 0), bool)
        hit_wall = z["contact_wall"] if "contact_wall" in z.files else np.zeros(len(t), bool)
        contact = z["contact"]
    keep = np.unique(np.searchsorted(t, np.arange(0, t[-1], STEP_S)))
    code = np.zeros(len(t), int)
    code[contact] = HIT["static"]                 # touching something: static unless shown otherwise below
    code[hit_wall] = HIT["wall"]
    if hit_obst.shape[1]:
        code[(hit_obst & ~np.asarray(moving, bool)).any(axis=1)] = HIT["static"]
        code[(hit_obst & np.asarray(moving, bool)).any(axis=1)] = HIT["mover"]   # a mover hit matters most
    worst = np.zeros(len(keep), int)
    for k, (a, b) in enumerate(zip(keep, list(keep[1:]) + [len(t)])):
        seg = code[a:b]
        worst[k] = HIT["mover"] if (seg == HIT["mover"]).any() else seg.max(initial=0)
    yaw = np.arctan2(obs[keep, 3], obs[keep, 4])
    run = {"x": np.round(base[keep, 0]*100).astype(int).tolist(), "y": np.round(base[keep, 1]*100).astype(int).tolist(),
           "yaw": np.round(yaw*100).astype(int).tolist(), "hit": worst.tolist()}
    if xy is not None:
        run["obst"] = np.round(xy[keep]*100).astype(int).reshape(len(keep), -1).tolist()
    return run


def load_model(name, folder):
    folder = ROOT/folder
    manifest = json.loads((folder/"manifest.json").read_text())
    expert = not (folder/"evaluation.json").exists()
    out = {}
    for e in manifest["episodes"]:
        ep = e["path"]
        moving = np.array(e.get("obstacle_is_moving", []), bool)
        hits = np.array(e.get("contact_steps_per_obstacle", []), bool)
        result = {"mover": bool((hits & moving).any()) if len(moving) else False,
                  "static": bool((hits & ~moving).any()) if len(moving) else False,
                  "wall": e.get("wall_contact_steps", 0) > 0}
        # every column, the expert's too, scored with the same base rule (eval_obstacles.score "parked")
        s = rescore_episode(folder/ep)
        why = ("" if s["parked"] else "touched something" if s["contact_steps"] else
               f"stopped {s['range_m']:.2f} m away, outside the reach area" if not s["in_reach_area"] else
               "not looking at the target" if s["worst_facing_error_rad"] > FACING_LIMIT else "did not stay still")
        result |= {"park": "parked" if s["parked"] else "no", "why": why, "range_m": round(s["range_m"], 2),
                   "path_m": round(s["base_path_length_m"], 1), "expert": expert,
                   "task": bool(e["success"]) if expert else None}
        out[ep] = {"result": result, "run": load_run(folder, ep, moving)}
    print(f"{name}: {len(out)} episodes from {folder.relative_to(ROOT)}")
    return out


def scenario_of(folder, ep):
    s = json.loads((ROOT/folder/ep/"episode.json").read_text())["scenario"]
    return {"room": s.get("room"), "target": [round(v, 2) for v in s["target"][:2]], "park": [round(v, 2) for v in s["park"]],
            "reach": [REACH_MIN_M, round(reach_limit(s["target"][2]), 3)],   # accepted parking distances round the target
            "start": [round(v, 2) for v in s["start"]],
            "obstacles": [{"r": round(o["r"], 3), "moving": bool(np.any(o.get("vel", [0, 0])))} for o in s["obstacles"]]}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("models", nargs="*", help="NAME=FOLDER, in display order")
    parser.add_argument("--out", default="runs/rooms-compare.html")
    args = parser.parse_args()
    models = [tuple(m.split("=", 1)) for m in args.models] or default_models()
    loaded = [load_model(name, folder) for name, folder in models]
    ids = sorted(set().union(*loaded))
    scenarios = {}
    for ep in ids:
        for (name, folder), runs in zip(models, loaded):
            if ep in runs:
                scenarios[ep] = scenario_of(folder, ep)
                break
    data = {"models": [name for name, _ in models], "folders": [folder for _, folder in models], "step": STEP_S,
            "scenarios": scenarios, "runs": {ep: [runs.get(ep) for runs in loaded] for ep in ids}}
    out = ROOT/args.out
    out.write_text(TEMPLATE.replace("__DATA__", json.dumps(data, separators=(",", ":"))))
    print(f"Wrote {out} ({len(ids)} scenarios, {len(models)} models, {out.stat().st_size/1e6:.1f} MB)")


TEMPLATE = r"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Room run comparison</title>
<style>
:root{--bg:#fafaf9;--panel:#fff;--ink:#1c1917;--mute:#78716c;--line:#e7e5e4;--grid:#f0efed;--wall:#44403c;
--main:#2563eb;--target:#dc2626;--park:#16a34a;--static:#a8a29e;--mover:#7c3aed;--hit:#dc2626;--touch:#ea580c;
--strict:#16a34a;--loose:#86efac;--no:#e7e5e4}
@media (prefers-color-scheme:dark){:root{--bg:#161514;--panel:#1f1e1d;--ink:#f5f5f4;--mute:#a8a29e;--line:#3a3836;
--grid:#262524;--wall:#d6d3d1;--main:#60a5fa;--target:#f87171;--park:#4ade80;--static:#78716c;--mover:#a78bfa;
--hit:#f87171;--touch:#fb923c;--strict:#22c55e;--loose:#166534;--no:#2e2c2a}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.4 system-ui,sans-serif}
main{max-width:1400px;margin:0 auto;padding:16px;display:grid;grid-template-columns:minmax(0,340px) minmax(0,1fr);gap:16px}
@media (max-width:900px){main{grid-template-columns:minmax(0,1fr)}}
h1{font-size:18px;margin:0 0 4px}.mute{color:var(--mute)}.panel{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px}
.scroll{overflow:auto;max-height:78vh}@media (max-width:900px){.scroll{max-height:40vh}}
table{border-collapse:collapse;font-variant-numeric:tabular-nums;font-size:12px}
th{font-weight:600;text-align:center;padding:2px 3px;position:sticky;top:0;background:var(--panel);z-index:1;vertical-align:bottom}
th.rot{height:120px;white-space:nowrap}th.rot div{writing-mode:vertical-rl;transform:rotate(180deg);margin:0 auto}
td{padding:1px}td.id{padding-right:6px;color:var(--mute);cursor:pointer}tr.on td.id{color:var(--ink);font-weight:600}
tr.on td{background:var(--grid)}
.cell{width:26px;height:18px;border-radius:4px;position:relative;cursor:pointer;margin:auto}
.cell i{position:absolute;right:2px;top:2px;width:6px;height:6px;border-radius:50%}
.parked{background:var(--strict)}.no{background:var(--no)}
tfoot td{font-weight:600;text-align:center;padding-top:4px;font-size:11px}
.legend{display:flex;flex-wrap:wrap;gap:4px 14px;margin:8px 0}.legend span{white-space:nowrap}
.sw{display:inline-block;width:12px;height:12px;border-radius:3px;margin-right:4px;vertical-align:-2px}
.dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:4px;vertical-align:0}
.row{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:0 0 12px}input[type=range]{flex:1;min-width:140px}
button.ctl,select.ctl{padding:5px 12px;border:1px solid var(--line);border-radius:6px;background:var(--panel);color:var(--ink);cursor:pointer;font:inherit}
#panels{display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:12px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:8px}
.card h2{font-size:13px;margin:0 0 2px}.card .res{font-size:12px;min-height:34px}
canvas{width:100%;aspect-ratio:1;display:block}
.ok{color:var(--strict)}.bad{color:var(--hit)}.warn{color:var(--touch)}#clock{font-variant-numeric:tabular-nums;min-width:64px}
</style></head><body><main>
<div>
 <h1>Room run comparison</h1>
 <div class="mute">Same 60 room validation scenarios for every model; arm held still, so learned models are scored on parking. Parked = stopped inside the arm's reach area, looking at the target (±30°), still for the last 2 s, no contact all run. Click a row or cell.</div>
 <div class="legend mute">
  <span><i class="sw parked"></i>parked</span><span><i class="sw no"></i>not parked</span>
  <span><i class="dot" style="background:var(--hit)"></i>hit a mover</span><span><i class="dot" style="background:var(--touch)"></i>hit wall / still obstacle</span>
 </div>
 <div class="mute" style="font-size:12px;margin-bottom:8px">Expert column: same base rule (its arm also reaches; whole-task result in the tooltip).</div>
 <div class="panel scroll"><table id="grid"></table></div>
</div>
<div>
 <div class="row"><button class="ctl" id="play">▶ Play</button>
  <select id="speed" class="ctl"><option value="1">1×</option><option value="3">3×</option><option value="6" selected>6×</option><option value="12">12×</option></select>
  <input type="range" id="time" min="0" step="0.1" value="0"><span id="clock" class="mute">0.0 s</span></div>
 <div class="legend mute">
  <span><i class="dot" style="background:var(--main)"></i>robot and path</span><span><i class="dot" style="background:var(--mover)"></i>moving obstacle</span>
  <span><i class="dot" style="background:var(--static)"></i>still obstacle</span><span><i class="dot" style="background:var(--target)"></i>target</span>
  <span><i class="dot" style="background:var(--park)"></i>reach area (where parking counts); circle = the expert's spot</span><span><i class="dot" style="background:var(--hit)"></i>contact point</span>
 </div>
 <div id="sel" style="font-weight:600;margin-bottom:8px"></div>
 <div id="panels"></div>
</div>
</main>
<script>
const D=__DATA__, ids=Object.keys(D.runs), M=D.models;
let cur=ids[0], t=0, playing=false, last=0;
const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const grid=document.getElementById('grid'), panels=document.getElementById('panels'), slider=document.getElementById('time');

// outcome grid
const short=id=>id.replace(/^[a-z]+_?/i,'');
let html='<thead><tr><th></th>'+M.map(m=>`<th class="rot"><div>${m}</div></th>`).join('')+'</tr></thead><tbody>';
for(const id of ids){html+=`<tr data-id="${id}"><td class="id">${short(id)}</td>`;
 D.runs[id].forEach((r,k)=>{if(!r){html+='<td></td>';return}const o=r.result;
  const dot=o.mover?'var(--hit)':(o.static||o.wall)?'var(--touch)':'';
  html+=`<td><div class="cell ${o.park}" title="${M[k]}: ${label(o)}">${dot?`<i style="background:${dot}"></i>`:''}</div></td>`});
 html+='</tr>'}
html+='</tbody><tfoot>';
const total=f=>M.map((_,k)=>ids.filter(id=>D.runs[id][k]&&f(D.runs[id][k].result)).length);
for(const [name,f] of [['parked',o=>o.park==='parked'],['no hit',o=>!o.mover&&!o.static&&!o.wall],['mover hit',o=>o.mover]])
 html+=`<tr><td class="id">${name}</td>${total(f).map(n=>`<td>${n}</td>`).join('')}</tr>`;
grid.innerHTML=html+'</tfoot>';
grid.querySelectorAll('tbody tr').forEach(tr=>tr.onclick=()=>select(tr.dataset.id));

function label(o){const park=o.park==='parked'?`parked ${o.range_m} m away`:`not parked: ${o.why}`;
 const hit=o.mover?'hit a mover':o.static?'hit a still obstacle':o.wall?'hit a wall':'no contact';
 return `${park} · ${hit}`+(o.expert?` · whole task ${o.task?'done':'failed'}`:'')}

// replay panels
const cards=M.map((m,k)=>{const c=document.createElement('div');c.className='card';
 c.innerHTML=`<h2>${m}</h2><div class="res mute"></div><canvas></canvas>`;panels.appendChild(c);
 return {res:c.querySelector('.res'),cv:c.querySelector('canvas')}});
function duration(){return Math.max(...D.runs[cur].map(r=>r?(r.run.x.length-1)*D.step:0))}
function select(id){cur=id;t=0;slider.max=duration().toFixed(1);
 grid.querySelectorAll('tbody tr').forEach(tr=>tr.classList.toggle('on',tr.dataset.id===id));
 const s=D.scenarios[id],nm=s.obstacles.filter(o=>o.moving).length;
 document.getElementById('sel').textContent=`${id} · ${s.obstacles.length} obstacles, ${nm} moving`;
 D.runs[id].forEach((r,k)=>{const o=r&&r.result;cards[k].res.innerHTML=!r?'not run':
  `<span class="${o.park==='parked'?'ok':'bad'}">${label(o)}</span>`+(o.path_m!==undefined?` · drove ${o.path_m} m`:'')});
 draw()}
function at(a,f){const i=Math.min(a.length-1,Math.floor(f)),j=Math.min(a.length-1,i+1),w=Math.min(1,Math.max(0,f-i));return a[i]+(a[j]-a[i])*w}
function draw(){const s=D.scenarios[cur],room=s.room||[-5,-5,5,5];
 document.getElementById('clock').textContent=t.toFixed(1)+' s';slider.value=t;
 D.runs[cur].forEach((r,k)=>{const cv=cards[k].cv,R=cv.getBoundingClientRect(),d=devicePixelRatio||1;
  cv.width=R.width*d;cv.height=R.width*d;const g=cv.getContext('2d');g.setTransform(d,0,0,d,0,0);const W=R.width;
  const pad=0.3,x0=room[0]-pad,y0=room[1]-pad,span=Math.max(room[2]-room[0],room[3]-room[1])+2*pad,sc=W/span;
  const P=(x,y)=>[(x-x0)*sc,W-(y-y0)*sc];
  g.strokeStyle=css('--grid');g.lineWidth=1;
  for(let v=Math.ceil(x0);v<x0+span;v++){const[a]=P(v,0);g.beginPath();g.moveTo(a,0);g.lineTo(a,W);g.stroke();
   const[,b]=P(0,v);g.beginPath();g.moveTo(0,b);g.lineTo(W,b);g.stroke()}
  const[ra,rb]=P(room[0],room[3]),[rc,rd]=P(room[2],room[1]);g.strokeStyle=css('--wall');g.lineWidth=3;g.strokeRect(ra,rb,rc-ra,rd-rb);
  // scoring: the base centre anywhere in this ring round the target (and facing it within 30 degrees)
  const[tx,ty]=P(...s.target);g.fillStyle=css('--park');g.globalAlpha=0.18;g.beginPath();
  g.arc(tx,ty,s.reach[1]*sc,0,2*Math.PI);g.arc(tx,ty,s.reach[0]*sc,0,2*Math.PI,true);g.fill('evenodd');
  g.globalAlpha=0.6;g.strokeStyle=css('--park');g.lineWidth=1;g.beginPath();g.arc(tx,ty,s.reach[1]*sc,0,7);g.stroke();
  g.globalAlpha=1;g.strokeStyle=css('--park');const[px,py]=P(...s.park);g.lineWidth=2;g.beginPath();g.arc(px,py,Math.max(4,0.1*sc),0,7);g.stroke();g.fillStyle=css('--target');g.beginPath();g.arc(tx,ty,4,0,7);g.fill();
  if(!r){g.fillStyle=css('--mute');g.fillText('not run',8,16);return}
  const u=r.run,n=u.x.length,f=Math.min(n-1,t/D.step),fi=Math.floor(f);
  // obstacles: still ones fixed, moving ones with a 2 s trail
  if(u.obst)s.obstacles.forEach((o,j)=>{const col=css(o.moving?'--mover':'--static');
   if(o.moving){g.strokeStyle=col;g.globalAlpha=0.35;g.lineWidth=2;g.beginPath();
    for(let q=Math.max(0,fi-10);q<=fi;q++){const[a,b]=P(u.obst[q][2*j]/100,u.obst[q][2*j+1]/100);q>Math.max(0,fi-10)?g.lineTo(a,b):g.moveTo(a,b)}g.stroke();g.globalAlpha=1}
   const[a,b]=P(at(u.obst.map(v=>v[2*j]),f)/100,at(u.obst.map(v=>v[2*j+1]),f)/100);g.fillStyle=col;g.beginPath();g.arc(a,b,o.r*sc,0,7);g.fill()});
  // path: whole route faint, travelled part solid, contact points marked
  const line=(upto,alpha,w)=>{g.globalAlpha=alpha;g.strokeStyle=css('--main');g.lineWidth=w;g.beginPath();
   for(let q=0;q<=upto;q++){const[a,b]=P(u.x[q]/100,u.y[q]/100);q?g.lineTo(a,b):g.moveTo(a,b)}g.stroke();g.globalAlpha=1};
  line(n-1,0.18,2);line(fi,1,2);
  for(let q=0;q<=fi;q++)if(u.hit[q]){const[a,b]=P(u.x[q]/100,u.y[q]/100);g.fillStyle=css(u.hit[q]===1?'--hit':'--touch');g.beginPath();g.arc(a,b,2.5,0,7);g.fill()}
  // robot: 0.7 x 0.5 m box with heading, red outline while touching something
  const[bx,by]=P(at(u.x,f)/100,at(u.y,f)/100),yaw=u.yaw[fi]/100,L=0.7*sc,Wd=0.5*sc,now=u.hit[fi];
  g.save();g.translate(bx,by);g.rotate(-yaw);g.strokeStyle=css(now===1?'--hit':now?'--touch':'--main');g.lineWidth=now?3:2;
  g.strokeRect(-L/2,-Wd/2,L,Wd);g.beginPath();g.moveTo(0,0);g.lineTo(L*0.8,0);g.stroke();g.restore();
  if(fi>=n-1){g.fillStyle=css('--mute');g.font='11px system-ui';g.fillText('run ended',6,W-6)}});
}
function tick(ts){if(playing){const dt=(ts-last)/1000*document.getElementById('speed').value;last=ts;
  t=Math.min(duration(),t+dt);if(t>=duration())setPlay(false);draw()}requestAnimationFrame(tick)}
function setPlay(p){playing=p;document.getElementById('play').textContent=p?'❚❚ Pause':'▶ Play';last=performance.now()}
document.getElementById('play').onclick=()=>{if(t>=duration())t=0;setPlay(!playing)};
slider.oninput=()=>{t=+slider.value;draw()};addEventListener('resize',draw);
matchMedia('(prefers-color-scheme: dark)').addEventListener('change',draw);
document.addEventListener('keydown',e=>{const k=ids.indexOf(cur);
 if(e.key==='ArrowDown'&&k<ids.length-1){select(ids[k+1]);e.preventDefault()}
 if(e.key==='ArrowUp'&&k>0){select(ids[k-1]);e.preventDefault()}
 if(e.key===' '){document.getElementById('play').click();e.preventDefault()}});
select(cur);requestAnimationFrame(tick);
</script></body></html>"""

if __name__ == "__main__":
    main()
