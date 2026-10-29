"""CLI: doctor -> collect -> train --model ... -> evaluate --model-file ..."""
import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
import joblib
import numpy as np
import sklearn
from models import ACTION_NAMES, FEATURE_NAMES, SCHEMA_VERSION, make_model, regression_metrics, stack_episode

ROOT = Path(__file__).resolve().parent

def read_json(path):
    return json.loads(Path(path).read_text())

def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2))

def settings(args):
    config = read_json(args.settings)
    if config["history"] < 1 or config["physics_steps_per_action"] < 1 or config["duration_s"] <= 0:
        raise ValueError("History, control interval and duration must be positive")
    return config

def scenarios(path):
    data = read_json(path)
    seen_ids, seen_targets = set(), set()
    for s in data:
        target = tuple(s["target"])
        if s["id"] in seen_ids or target in seen_targets:
            raise ValueError("Each scenario needs a unique ID and target; avoid cross-split duplicates")
        if s["split"] not in ("train", "validation", "test") or len(target)!=3 or not np.isfinite(target).all():
            raise ValueError("Invalid scenario split or target")
        if not s["id"].replace("_", "").replace("-", "").isalnum():
            raise ValueError("Scenario IDs must contain only letters, numbers, underscores or hyphens")
        seen_ids.add(s["id"])
        seen_targets.add(target)
    return data

def environment_record():
    return {"python":sys.version,"executable":sys.executable,"platform":platform.platform(),
            "numpy":np.__version__,"sklearn":sklearn.__version__}

def doctor(args):
    config = settings(args)
    from robot_env import provenance
    import pinocchio, pybullet, simulation_and_control
    assets = Path(config["roboenv_root"])/"models/summit_xl_arm_description"
    if not assets.is_dir():
        raise FileNotFoundError(assets)
    print(json.dumps({"imports":"OK", "assets":str(assets),"features":len(FEATURE_NAMES),
        "history":config["history"],"model_inputs":len(FEATURE_NAMES)*config["history"],
        "outputs":len(ACTION_NAMES),"source_sha256":provenance(config),
        "environment":environment_record()},indent=2))

def collect(args):
    from robot_env import rollout
    config = settings(args)
    selected = scenarios(args.scenarios)
    if args.only:
        selected = [s for s in selected if s["id"] in args.only]
        if set(args.only)!={s["id"] for s in selected}:
            raise ValueError("An --only scenario does not exist")
    out = Path(args.out).resolve()
    if out.exists():
        raise FileExistsError(f"Choose a new dataset directory: {out}")
    out.mkdir(parents=True)
    manifest = {"schema_version":SCHEMA_VERSION,"feature_names":FEATURE_NAMES,"action_names":ACTION_NAMES,
                "settings":config,"scenarios":selected,"environment":environment_record(),"episodes":[]}
    write_json(out/"manifest.json",manifest)
    for scenario in selected:
        print(f"Collecting {scenario['id']} ({scenario['split']}) ...",flush=True)
        summary = rollout(config,scenario,out/scenario["id"],gui=args.gui)
        manifest["episodes"].append({"path":scenario["id"],**summary})
        write_json(out/"manifest.json",manifest)
        print(f"  {summary['termination']}: error={summary['final_error_m']:.4f} m, "
              f"time={summary['simulation_time_s']:.2f} s",flush=True)
    print(f"Dataset saved: {out}")

def load_split(folder, manifest, split, history):
    xs, ys, used = [], [], []
    for episode in manifest["episodes"]:
        if episode["split"] != split or not episode["success"]:
            continue
        with np.load(folder/episode["path"]/"trajectory.npz",allow_pickle=False) as z:
            x,y=z["observations"],z["actions"]
        if x.shape[1]!=len(FEATURE_NAMES) or y.shape!=(len(x),len(ACTION_NAMES)):
            raise ValueError("Dataset dimensions do not match observation/action schema")
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("Dataset contains nonfinite values")
        xs.append(stack_episode(x,history)); ys.append(y); used.append(episode["scenario_id"])
    if not xs:
        raise ValueError(f"No successful expert episodes in {split}; inspect collection summaries before training")
    return np.concatenate(xs),np.concatenate(ys),used

def train(args):
    data = Path(args.data).resolve()
    manifest = read_json(data/"manifest.json")
    config = manifest["settings"]
    # Model hyperparameters may change, but simulator/observation settings remain those used for collection.
    overrides = settings(args)
    config = {**config, "models":overrides["models"], "seed":overrides["seed"]}
    if manifest["schema_version"] != SCHEMA_VERSION or manifest["feature_names"] != FEATURE_NAMES:
        raise ValueError("Incompatible dataset schema")
    out = Path(args.out).resolve()
    if out.exists():
        raise FileExistsError(f"Choose a new model directory: {out}")
    x,y,train_ids = load_split(data,manifest,"train",config["history"])
    xv,yv,val_ids = load_split(data,manifest,"validation",config["history"])
    print(f"Training {args.model}: {len(x)} samples across {len(train_ids)} episodes; "
          f"validation={len(val_ids)} separate episodes",flush=True)
    estimator = make_model(args.model,config)
    start = time.perf_counter()
    estimator.fit(x,y)
    training_time = time.perf_counter()-start
    metrics = {"model":args.model,"training_seconds":training_time,"training_episodes":train_ids,
        "validation_episodes":val_ids,"training_samples":len(x),"validation_samples":len(xv),
        "training":regression_metrics(y,estimator.predict(x)),
        "validation":regression_metrics(yv,estimator.predict(xv)),
        "note":"Offline errors are diagnostics, not evidence of closed-loop task success. Test data were not loaded."}
    out.mkdir(parents=True)
    joblib.dump({"estimator":estimator,"settings":config,"schema_version":SCHEMA_VERSION,
        "feature_names":FEATURE_NAMES,"action_names":ACTION_NAMES,"model_name":args.model,
        "dataset_manifest_sha256":hashlib.sha256((data/"manifest.json").read_bytes()).hexdigest(),
        "training_ids":train_ids,
        "training_targets":[s["target"] for s in manifest["scenarios"] if s["id"] in train_ids],
        "environment":environment_record()},out/"model.joblib")
    write_json(out/"training.json",metrics)
    print(json.dumps(metrics,indent=2))
    print(f"Model saved: {out/'model.joblib'}")

def evaluate(args):
    from robot_env import rollout
    model_path = Path(args.model_file).resolve()
    # Pickle/joblib is only for locally created, trusted artifacts. Never load a downloaded untrusted model.
    bundle = joblib.load(model_path)
    if bundle["schema_version"]!=SCHEMA_VERSION or bundle["feature_names"]!=FEATURE_NAMES:
        raise ValueError("Incompatible model schema")
    config = bundle["settings"]
    selected = [s for s in scenarios(args.scenarios) if s["split"]==args.split]
    if not selected:
        raise ValueError("No scenarios in selected evaluation split")
    if any(s["id"] in bundle["training_ids"] for s in selected):
        raise ValueError("Evaluation scenario overlaps model training episodes")
    if any(s["target"] in bundle["training_targets"] for s in selected):
        raise ValueError("Evaluation target overlaps training, even if its ID was renamed")
    out = Path(args.out).resolve()
    if out.exists():
        raise FileExistsError(f"Choose a new evaluation directory: {out}")
    out.mkdir(parents=True)
    results = []
    for scenario in selected:
        print(f"Evaluating {scenario['id']} ...",flush=True)
        result = rollout(config,scenario,out/scenario["id"],bundle["estimator"],gui=args.gui)
        results.append(result)
        print(f"  {result['termination']}: final error {result['final_error_m']:.4f} m",flush=True)
    with (out/"episodes.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(results[0]));writer.writeheader();writer.writerows(results)
    summary={"model":str(model_path),"model_sha256":hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "split":args.split,"episodes":len(results),"success_rate":float(np.mean([r["success"] for r in results])),
        "mean_final_error_m":float(np.mean([r["final_error_m"] for r in results])),
        "settings":config,"environment":environment_record(),
        "note":"Small pilot suite, fixed initial pose, no obstacles. Not a generalisation or robustness study."}
    write_json(out/"evaluation.json",summary)
    plot_results(out,results)
    print(json.dumps(summary,indent=2))

def plot_results(folder, results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,4))
    for r in results:
        with np.load(folder/r["scenario_id"]/"trajectory.npz") as z:
            ax.plot(z["time_s"],z["error_m"],label=r["scenario_id"])
    ax.set(xlabel="Simulation time (s)",ylabel="End-effector error (m)")
    ax.legend();fig.tight_layout();fig.savefig(folder/"error.png",dpi=140);plt.close(fig)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings",default=str(ROOT/"settings.json"))
    sub=parser.add_subparsers(dest="command",required=True)
    sub.add_parser("doctor").set_defaults(func=doctor)
    p=sub.add_parser("collect")
    p.add_argument("--scenarios",default=str(ROOT/"scenarios.json"));p.add_argument("--out",required=True)
    p.add_argument("--only",nargs="+");p.add_argument("--gui",action="store_true");p.set_defaults(func=collect)
    p=sub.add_parser("train")
    p.add_argument("--data",required=True);p.add_argument("--model",choices=["ridge","mlp","forest"],default="mlp")
    p.add_argument("--out",required=True);p.set_defaults(func=train)
    p=sub.add_parser("evaluate")
    p.add_argument("--model-file",required=True);p.add_argument("--scenarios",default=str(ROOT/"scenarios.json"))
    p.add_argument("--split",choices=["validation","test"],default="validation")
    p.add_argument("--out",required=True);p.add_argument("--gui",action="store_true");p.set_defaults(func=evaluate)
    args=parser.parse_args()
    args.func(args)

if __name__=="__main__":
    main()
