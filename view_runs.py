"""Build a replay page (viewer.html) for a run folder: a top-down animation of every episode.

It shows the base path and heading, the hand, the target and the expert's parking spot, plus end-effector
error over time. With --expert, the expert's run on the same target is overlaid for comparison.

Examples:
    python view_runs.py archive/runs/mlp-v3-validation --expert runs/dataset-v3
    python view_runs.py runs/dataset-v3 --split validation
Then open the printed viewer.html in a browser.
"""
import argparse
import json
from pathlib import Path
import numpy as np

STEP_S = 0.02  # replay sample period; the 1 kHz logs are thinned to 50 Hz

def parking_spot(target):
    """The expert's base goal for this target (same rule as robot_env.py), drawn for reference only."""
    standoff = np.interp(target[2], [0.1, 0.5, 1.1, 1.25], [0.80, 0.65, 0.65, 0.35])
    bearing = np.arctan2(target[1], target[0]) if np.linalg.norm(target[:2]) > 1e-4 else 0.0
    return (target[:2] - standoff*np.array([np.cos(bearing), np.sin(bearing)])).tolist()

def load_episode(folder):
    with np.load(folder/"trajectory.npz", allow_pickle=False) as z:
        t, obs, base, ee, err, act = z["time_s"], z["observations"], z["base_m"], z["ee_m"], z["error_m"], z["actions"]
        lidar = z["lidar"] if "lidar" in z.files else None
    keep = np.unique(np.searchsorted(t, np.arange(0, t[-1], STEP_S)))
    r = lambda a: np.round(a[keep], 3).tolist()
    summary = json.loads((folder/"summary.json").read_text())
    target = obs[0, 5:8]
    scenario = json.loads((folder/"episode.json").read_text())["scenario"] if (folder/"episode.json").exists() else {}
    extra = {}
    if "obstacles" in scenario:   # obstacle runs: their own parking spot, obstacles and 10 Hz lidar (in cm)
        extra = {"park": scenario["park"], "obstacles": scenario["obstacles"]}
    if lidar is not None:
        extra["lidar"] = np.round(lidar[keep[::5]]*100).astype(int).tolist()
    return {"t": r(t), "x": r(base[:, 0]), "y": r(base[:, 1]), "z": r(base[:, 2]),
            "yaw": r(np.arctan2(obs[:, 3], obs[:, 4])), "ex": r(ee[:, 0]), "ey": r(ee[:, 1]),
            "err": r(err), "wheels": np.round(act[keep, :4], 2).tolist(),
            "target": target.round(3).tolist(), "park": np.round(parking_spot(target), 3).tolist(),
            "success": summary["success"], "termination": summary["termination"],
            "final_error": round(summary["final_error_m"], 3), "controller": summary["controller"]} | extra

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", help="a dataset or evaluation folder containing episode subfolders")
    parser.add_argument("--expert", help="dataset folder holding the expert runs on the same scenario IDs")
    parser.add_argument("--split", help="only episodes whose ID starts with this, e.g. validation")
    args = parser.parse_args()
    run = Path(args.run)
    episodes = {}
    for folder in sorted(p for p in run.iterdir() if (p/"trajectory.npz").exists()):
        if args.split and not folder.name.startswith(args.split):
            continue
        episodes[folder.name] = {"main": load_episode(folder)}
        if args.expert and (Path(args.expert)/folder.name/"trajectory.npz").exists():
            episodes[folder.name]["expert"] = load_episode(Path(args.expert)/folder.name)
    if not episodes:
        raise ValueError(f"No episodes with trajectory.npz in {run}")
    page = TEMPLATE.replace("__TITLE__", run.name).replace("__DATA__", json.dumps(episodes))
    out = run/"viewer.html"
    out.write_text(page)
    print(f"Wrote {out.resolve()} ({len(episodes)} episodes)")

TEMPLATE = r"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Run viewer: __TITLE__</title>
<style>
:root{--bg:#fafaf9;--panel:#fff;--ink:#1c1917;--mute:#78716c;--line:#e7e5e4;--grid:#f0efed;
--main:#2563eb;--expert:#9ca3af;--target:#dc2626;--park:#16a34a;--hand:#d97706;--obst:#78716c;--ray:rgba(220,38,38,.18)}
@media (prefers-color-scheme:dark){:root{--bg:#161514;--panel:#1f1e1d;--ink:#f5f5f4;--mute:#a8a29e;--line:#3a3836;
--grid:#262524;--main:#60a5fa;--expert:#6b7280;--target:#f87171;--park:#4ade80;--hand:#fbbf24;--obst:#a8a29e;--ray:rgba(248,113,113,.22)}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.4 system-ui,sans-serif}
main{max-width:1180px;margin:0 auto;padding:16px;display:grid;grid-template-columns:220px 1fr;gap:16px}
@media (max-width:760px){main{grid-template-columns:1fr}}
h1{font-size:18px;margin:0 0 4px}.mute{color:var(--mute)}.panel{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px}
#list button{display:flex;justify-content:space-between;width:100%;text-align:left;padding:6px 8px;margin:2px 0;border:1px solid transparent;
border-radius:6px;background:none;color:var(--ink);cursor:pointer;font:inherit}#list button.on{border-color:var(--main)}
.ok{color:var(--park)}.bad{color:var(--target)}canvas{width:100%;display:block;border-radius:8px;background:var(--panel)}
.row{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:10px 0}input[type=range]{flex:1;min-width:160px}
button.ctl{padding:5px 12px;border:1px solid var(--line);border-radius:6px;background:var(--panel);color:var(--ink);cursor:pointer;font:inherit}
.legend span{margin-right:14px;white-space:nowrap}.dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:4px;vertical-align:-1px}
#info{font-variant-numeric:tabular-nums}
</style></head><body><main>
<div><h1>__TITLE__</h1><div class="mute">Click an episode</div><div id="list" class="panel" style="margin-top:8px"></div></div>
<div>
 <div class="legend mute"><span><i class="dot" style="background:var(--main)"></i>this run's base</span>
 <span><i class="dot" style="background:var(--expert)"></i>expert's base</span><span><i class="dot" style="background:var(--hand)"></i>hand</span>
 <span><i class="dot" style="background:var(--target)"></i>target</span><span><i class="dot" style="background:var(--park)"></i>expert parking spot</span>
 <span><i class="dot" style="background:var(--obst)"></i>obstacle</span><span><i class="dot" style="background:var(--target);opacity:.4"></i>lidar rays</span></div>
 <canvas id="map" height="520"></canvas>
 <div class="row"><button class="ctl" id="play">▶ Play</button>
  <select id="speed" class="ctl"><option value="1">1×</option><option value="2" selected>2×</option><option value="5">5×</option></select>
  <input type="range" id="time" min="0" value="0"><button class="ctl" id="fit">Zoom to all</button></div>
 <div id="info" class="panel"></div>
 <canvas id="err" height="160" style="margin-top:12px"></canvas>
</div></main>
<script>
const DATA=__DATA__, ids=Object.keys(DATA);
let cur=ids[0], i=0, playing=false, fitAll=false, last=0;
const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const map=document.getElementById('map'), errc=document.getElementById('err'), slider=document.getElementById('time');
const list=document.getElementById('list');
ids.forEach(id=>{const e=DATA[id].main,b=document.createElement('button');
 b.innerHTML=`<span>${id}</span><span class="${e.success?'ok':'bad'}">${e.success?'✓':'✗'} ${e.final_error} m</span>`;
 b.onclick=()=>{cur=id;i=0;render()};b.dataset.id=id;list.appendChild(b)});
function fitCanvas(c){// keep the CSS height separate: assigning c.height also rewrites the height attribute
 const h=+(c.dataset.h||(c.dataset.h=c.getAttribute('height'))),r=c.getBoundingClientRect(),d=devicePixelRatio||1;
 c.width=r.width*d;c.height=h*d;c.style.height=h+'px';const g=c.getContext('2d');g.setTransform(d,0,0,d,0,0);return [g,r.width,h]}
function view(ep){const m=ep.main;let xs=[m.x[0],m.target[0],m.park[0]],ys=[m.y[0],m.target[1],m.park[1]];
 for(const o of m.obstacles||[]){xs.push(o.xy[0]-o.r,o.xy[0]+o.r);ys.push(o.xy[1]-o.r,o.xy[1]+o.r)}
 if(fitAll)for(const k of['main','expert'])if(ep[k]){xs=xs.concat(ep[k].x);ys=ys.concat(ep[k].y)}
 let x0=Math.min(...xs)-0.6,x1=Math.max(...xs)+0.6,y0=Math.min(...ys)-0.6,y1=Math.max(...ys)+0.6;return {x0,x1,y0,y1}}
function draw(){const ep=DATA[cur],m=ep.main;const [g,W,H]=fitCanvas(map);
 const v=view(ep),s=Math.min(W/(v.x1-v.x0),H/(v.y1-v.y0)),cx=(v.x0+v.x1)/2,cy=(v.y0+v.y1)/2;
 const P=(x,y)=>[W/2+(x-cx)*s,H/2-(y-cy)*s];
 // 0.5 m grid
 g.strokeStyle=css('--grid');g.lineWidth=1;const st=s<40?5:0.5;
 for(let x=Math.ceil(v.x0/st)*st;x<v.x1;x+=st){const[a]=P(x,0);g.beginPath();g.moveTo(a,0);g.lineTo(a,H);g.stroke()}
 for(let y=Math.ceil(v.y0/st)*st;y<v.y1;y+=st){const[,b]=P(0,y);g.beginPath();g.moveTo(0,b);g.lineTo(W,b);g.stroke()}
 g.fillStyle=css('--mute');g.font='12px system-ui';g.fillText(`grid ${st} m`,8,H-8);
 const path=(e,col,upto,w)=>{g.strokeStyle=col;g.lineWidth=w;g.beginPath();for(let k=0;k<=Math.min(upto,e.x.length-1);k++){const[a,b]=P(e.x[k],e.y[k]);k?g.lineTo(a,b):g.moveTo(a,b)}g.stroke()};
 const t=m.t[i];
 if(ep.expert){const e=ep.expert,k=Math.min(e.t.length-1,e.t.findIndex(v=>v>=t)<0?e.t.length-1:e.t.findIndex(v=>v>=t));
  path(e,css('--expert'),e.x.length,2);robot(g,P,e,k,css('--expert'),s)}
 g.fillStyle=css('--obst');for(const o of m.obstacles||[]){const[a,b]=P(...o.xy);g.beginPath();g.arc(a,b,o.r*s,0,7);g.fill()}
 if(m.lidar){const L=m.lidar[Math.min(m.lidar.length-1,Math.floor(i/5))],n=L.length;g.strokeStyle=css('--ray');g.lineWidth=1;
  for(let k=0;k<n;k++){const a=m.yaw[i]+2*Math.PI*k/n,d=L[k]/100,[x0,y0]=P(m.x[i],m.y[i]),[x1,y1]=P(m.x[i]+d*Math.cos(a),m.y[i]+d*Math.sin(a));
   g.beginPath();g.moveTo(x0,y0);g.lineTo(x1,y1);g.stroke()}}
 g.globalAlpha=0.3;path(m,css('--main'),m.x.length,2);g.globalAlpha=1;  // whole route, faint
 path(m,css('--main'),i,3);
 const[px,py]=P(...m.park);g.strokeStyle=css('--park');g.lineWidth=2;g.beginPath();g.arc(px,py,Math.max(4,0.08*s),0,7);g.stroke();
 const[tx,ty]=P(m.target[0],m.target[1]);g.fillStyle=css('--target');g.beginPath();g.arc(tx,ty,6,0,7);g.fill();
 robot(g,P,m,i,css('--main'),s);
 const[hx,hy]=P(m.ex[i],m.ey[i]);g.fillStyle=css('--hand');g.beginPath();g.arc(hx,hy,4,0,7);g.fill();
 drawErr();info()}
function robot(g,P,e,k,col,s){const[a,b]=P(e.x[k],e.y[k]),yaw=e.yaw[k],L=Math.max(10,0.35*s),Wd=Math.max(7,0.25*s);
 g.save();g.translate(a,b);g.rotate(-yaw);g.strokeStyle=col;g.lineWidth=2;g.strokeRect(-L/2,-Wd/2,L,Wd);
 g.beginPath();g.moveTo(0,0);g.lineTo(L*0.9,0);g.stroke();g.restore()}
function drawErr(){const ep=DATA[cur],m=ep.main;const[g,W,H]=fitCanvas(errc);const T=Math.max(m.t.at(-1),ep.expert?ep.expert.t.at(-1):0);
 const cap=Math.max(2,Math.min(5,Math.max(...m.err.slice(0,50)))*1.2);const X=t=>40+(W-50)*t/T,Y=e=>H-20-(H-30)*Math.min(e,cap)/cap;
 g.strokeStyle=css('--line');g.strokeRect(40,10,W-50,H-30);g.fillStyle=css('--mute');g.font='11px system-ui';
 g.fillText('hand error (m)',44,22);g.fillText(cap.toFixed(1),8,16);g.fillText('0',26,H-18);g.fillText(T.toFixed(0)+' s',W-34,H-4);
 const line=(e,col)=>{g.strokeStyle=col;g.lineWidth=2;g.beginPath();e.t.forEach((t,k)=>{k?g.lineTo(X(t),Y(e.err[k])):g.moveTo(X(t),Y(e.err[k]))});g.stroke()};
 if(ep.expert)line(ep.expert,css('--expert'));line(m,css('--main'));
 g.strokeStyle=css('--ink');g.beginPath();g.moveTo(X(m.t[i]),10);g.lineTo(X(m.t[i]),H-20);g.stroke()}
function phase(w){const l=(w[1]+w[2])/2,r=(w[0]+w[3])/2;if(Math.abs(l)+Math.abs(r)<0.05)return 'stopped';
 if(Math.sign(l)!==Math.sign(r)&&Math.abs(l+r)<0.5*Math.abs(l-r))return 'turning on the spot';return Math.abs(l-r)<0.3?'driving straight':'driving and turning'}
function info(){const ep=DATA[cur],m=ep.main,d=Math.hypot(m.x[i]-m.park[0],m.y[i]-m.park[1]);
 document.querySelectorAll('#list button').forEach(b=>b.classList.toggle('on',b.dataset.id===cur));
 slider.max=m.t.length-1;slider.value=i;
 document.getElementById('info').innerHTML=`<b>${cur}</b> · ${m.controller} · <span class="${m.success?'ok':'bad'}">${m.termination}</span><br>
 t = ${m.t[i].toFixed(2)} s · hand error <b>${m.err[i].toFixed(3)} m</b> · base ${d.toFixed(2)} m from parking spot · base height ${m.z[i].toFixed(2)} m<br>
 wheels FR/FL/BL/BR = ${m.wheels[i].map(w=>w.toFixed(1)).join(' / ')} rad/s → <b>${phase(m.wheels[i])}</b>`}
function render(){draw()}
function tick(ts){if(playing){const m=DATA[cur].main;const dt=(ts-last)/1000*document.getElementById('speed').value;
 if(dt>=0.02){i=Math.min(m.t.length-1,i+Math.round(dt/0.02));last=ts;if(i===m.t.length-1)setPlay(false);draw()}}requestAnimationFrame(tick)}
function setPlay(p){playing=p;document.getElementById('play').textContent=p?'❚❚ Pause':'▶ Play';last=performance.now()}
document.getElementById('play').onclick=()=>{if(i>=DATA[cur].main.t.length-1)i=0;setPlay(!playing)};
document.getElementById('fit').onclick=e=>{fitAll=!fitAll;e.target.textContent=fitAll?'Zoom to target':'Zoom to all';draw()};
slider.oninput=()=>{i=+slider.value;draw()};addEventListener('resize',draw);
matchMedia('(prefers-color-scheme: dark)').addEventListener('change',draw);
render();requestAnimationFrame(tick);
</script></body></html>"""

if __name__ == "__main__":
    main()
