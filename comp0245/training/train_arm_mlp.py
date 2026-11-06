"""Train the arm MLP on the reaching phases of the expert dataset: 24 arm inputs -> 7 joint
torques, on top of gravity.

    python -m comp0245.training.train_arm_mlp --check              check the data and show the \\
        torque limits; trains nothing
    python -m comp0245.training.train_arm_mlp --out runs/arm1      train; saves best.pt, \\
        metrics.csv, curves.png, result.json

    torque = clip(gravity_torque(q) + network(inputs), -limit, +limit)

A reaching row lies between the base stopping at a dot and that dot being touched. Each joint has
its own torque limit:
LIMIT_MARGIN x its 99.9th percentile on the training reach rows, at most ARM_MAX_NM. Expert
labels are clipped to these limits, and so is the MLP's torque when it is scored and when it
drives. The test split is never read here.
"""

import argparse
import csv
import glob
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from comp0245.arm_data import gravity_torque
from comp0245.models import ARM_INPUT_NAMES, arm_inputs
from comp0245.training.train_mlp import CASES, verdict

ARM_MAX_NM = 30.0  # settings.json arm_limit_nm: what the simulator run allows anyway
LIMIT_MARGIN = 1.25
JOINTS = range(7)
COLUMNS = [
    "epoch",
    "train_loss",
    "train_check_loss",
    "val_loss",
    "val_r2",
    "val_rmse_nm",
    *[f"val_rmse_joint{j}" for j in JOINTS],
    *[f"val_rmse_{c}" for c in CASES],
    "val_clipped_share",
    "learning_rate",
    "seconds",
]


def reach_rows(t, wheels, picks):
    """Row indices of every reaching phase of a run: walk back from each touch while the
    expert's wheels are exactly zero."""
    rows = []
    stopped = np.all(wheels == 0, axis=1)
    for pick in picks:
        end = int(np.searchsorted(t, pick["time_s"]))
        start = end
        while start > 0 and stopped[start - 1]:
            start -= 1
        rows.append(np.arange(start, end))
    return np.concatenate(rows) if rows else np.zeros(0, int)


def load(folder, split):
    """Inputs (rows, 24), expert torques (rows, 7), gravity (rows, 7), case (rows,), and reach
    seconds per dot."""
    xs, torques, gravity, cases, reach_s = [], [], [], [], []
    runs = 0
    for name in sorted(glob.glob(f"{folder}/case*/part*/*/trajectory.npz")):
        run = Path(name).parent
        scenario = json.loads((run / "episode.json").read_text())["scenario"]
        summary = json.loads((run / "summary.json").read_text())
        if scenario["split"] != split or not summary["success"]:
            continue
        z = np.load(name)
        obs, ee, t, actions = z["observations"], z["ee_m"], z["time_s"], z["actions"]
        for pick in summary["picks"]:
            one = reach_rows(t, actions[:, :4], [pick])
            reach_s.append(len(one) * float(np.median(np.diff(t))))
        rows = reach_rows(t, actions[:, :4], summary["picks"])
        if not len(rows):
            continue
        xs.append(np.array([arm_inputs(obs[i], ee[i]) for i in rows], np.float32))
        torques.append(actions[rows, 4:11].astype(np.float32))
        gravity.append(np.array([gravity_torque(obs[i, 8:15]) for i in rows], np.float32))
        cases.append(np.full(len(rows), CASES.index(scenario["case"]), np.int8))
        runs += 1
    if not runs:
        raise SystemExit(f"No {split} runs found in {folder}")
    return {
        "x": np.concatenate(xs),
        "torque": np.concatenate(torques),
        "gravity": np.concatenate(gravity),
        "case": np.concatenate(cases),
        "reach_s": np.array(reach_s),
        "runs": runs,
    }


def torque_limits(torque):
    return np.minimum(
        LIMIT_MARGIN * np.percentile(np.abs(torque), 99.9, axis=0), ARM_MAX_NM
    ).astype(np.float32)


def make_arm(hidden, layers):
    parts, size = [], len(ARM_INPUT_NAMES)
    for _ in range(layers):
        parts += [nn.Linear(size, hidden), nn.ReLU()]
        size = hidden
    return nn.Sequential(*parts, nn.Linear(size, 7))


def load_arm(path):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if list(saved["input_names"]) != ARM_INPUT_NAMES:
        raise ValueError(f"{path}: inputs do not match models.ARM_INPUT_NAMES")
    model = make_arm(saved["hidden"], saved["layers"])
    model.load_state_dict(saved["state_dict"])
    return model.eval(), saved


@torch.no_grad()
def network_output(model, x, s, batch=16384):
    model.eval()
    scaled = (x - s["x_mean"]) / s["x_std"]
    return (
        np.concatenate(
            [
                model(torch.from_numpy(scaled[i : i + batch])).numpy()
                for i in range(0, len(x), batch)
            ]
        )
        * s["y_std"]
        + s["y_mean"]
    )


def plot(rows, path):
    epochs = [r["epoch"] for r in rows]
    figure, (a, b) = plt.subplots(1, 2, figsize=(11, 4))
    a.plot(epochs, [r["train_check_loss"] for r in rows], label="train")
    a.plot(epochs, [r["val_loss"] for r in rows], label="validation")
    a.set(
        xlabel="epoch", ylabel="scaled MSE (1 = predicts the average)", yscale="log", title="Loss"
    )
    a.legend()
    for j in JOINTS:
        b.plot(epochs, [r[f"val_rmse_joint{j}"] for r in rows], label=f"joint {j}")
    b.set(xlabel="epoch", ylabel="torque RMSE (N m)", title="Validation error per joint")
    b.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def fit(train, val, out, args):
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    rng = np.random.default_rng(args.seed)
    limit = torque_limits(train["torque"])
    y = (
        np.clip(train["torque"], -limit, limit) - train["gravity"]
    )  # the network learns what gravity does not explain
    val_y = np.clip(val["torque"], -limit, limit)
    s = {
        "x_mean": train["x"].mean(axis=0),
        "x_std": train["x"].std(axis=0) + 1e-6,
        "y_mean": y.mean(axis=0),
        "y_std": y.std(axis=0) + 1e-6,
    }
    X, Y = (
        torch.from_numpy((train["x"] - s["x_mean"]) / s["x_std"]),
        torch.from_numpy((y - s["y_mean"]) / s["y_std"]),
    )
    model = make_arm(args.hidden, args.layers)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, factor=0.5, patience=args.lr_patience, threshold=0.001
    )
    loss_fn = nn.MSELoss()
    check = rng.choice(len(X), min(len(val_y), len(X)), replace=False)
    val_scaled = (val_y - val["gravity"] - s["y_mean"]) / s["y_std"]
    print(
        f"train {train['runs']} runs, {len(X)} rows; validation {val['runs']} runs, {len(val_y)} "
        "rows; "
        f"{sum(p.numel() for p in model.parameters())} parameters; {args.threads} threads\n"
        f"torque limits N m: {np.round(limit, 2).tolist()}",
        flush=True,
    )
    rows, best_loss, stale = [], float("inf"), 0
    for epoch in range(1, args.epochs + 1):
        begin = time.time()
        model.train()
        total, order = 0.0, torch.randperm(len(X))
        for i in range(0, len(X), args.batch):
            idx = order[i : i + args.batch]
            loss = loss_fn(model(X[idx]), Y[idx])
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            total += loss.item() * len(idx)
        out_val = network_output(model, val["x"], s)
        val_loss = float(np.mean(((out_val - s["y_mean"]) / s["y_std"] - val_scaled) ** 2))
        if not np.isfinite(val_loss):
            raise SystemExit(f"Stopped: the validation loss became {val_loss} at epoch {epoch}")
        unclipped = val["gravity"] + out_val
        torque = np.clip(unclipped, -limit, limit)
        out_check = network_output(model, train["x"][check], s)
        row = {
            "epoch": epoch,
            "train_loss": total / len(X),
            "train_check_loss": float(np.mean(((out_check - y[check]) / s["y_std"]) ** 2)),
            "val_loss": val_loss,
            "val_r2": 1 - val_loss,
            "val_rmse_nm": float(np.sqrt(np.mean((torque - val_y) ** 2))),
            **{
                f"val_rmse_joint{j}": float(np.sqrt(np.mean((torque[:, j] - val_y[:, j]) ** 2)))
                for j in JOINTS
            },
            **{
                f"val_rmse_{c}": float(
                    np.sqrt(np.mean((torque[val["case"] == k] - val_y[val["case"] == k]) ** 2))
                )
                if (val["case"] == k).any()
                else float("nan")
                for k, c in enumerate(CASES)
            },
            "val_clipped_share": float(np.mean(np.abs(unclipped) > limit)),
            "learning_rate": optimiser.param_groups[0]["lr"],
            "seconds": time.time() - begin,
        }
        rows.append(row)
        scheduler.step(val_loss)
        improved = val_loss < (1 - 0.001) * best_loss
        stale = 0 if improved else stale + 1
        if improved:
            best_loss = val_loss
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "hidden": args.hidden,
                    "layers": args.layers,
                    "epoch": epoch,
                    "input_names": ARM_INPUT_NAMES,
                    "limit": limit,
                    **s,
                },
                out / "best.pt",
            )
        with (out / "metrics.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        plot(rows, out / "curves.png")
        print(
            f"epoch {epoch:3d}  train {row['train_check_loss']:.4f}  val {val_loss:.4f}  R2 "
            f"{row['val_r2']:.3f}  RMSE {row['val_rmse_nm']:.3f} N m  "
            f"beyond limits {100 * row['val_clipped_share']:.2f}%  lr {row['learning_rate']:.1e}  "
            f"{row['seconds']:.0f} s{'  *best' if improved else ''}",
            flush=True,
        )
        if stale >= args.patience:
            print(f"Early stop: no improvement for {stale} epochs", flush=True)
            break
    label, why = verdict(rows)
    best = min(rows, key=lambda r: r["val_loss"])
    (out / "result.json").write_text(
        json.dumps(
            {
                "verdict": label,
                "why": why,
                "best": best,
                "epochs_run": len(rows),
                "limit": limit.tolist(),
                "settings": vars(args),
            },
            indent=1,
        )
    )
    print(f"\nVERDICT: {label}\n{why}")
    print(
        "Best epoch RMSE per joint, N m: "
        + ", ".join(f"{j}: {best[f'val_rmse_joint{j}']:.3f}" for j in JOINTS)
    )
    print(
        f"Saved {out / 'best.pt'}. Whether the hand reaches the dots takes the closed-loop test: "
        "comp0245/evaluation/eval_arm_mlp.py"
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", default="runs/dataset")
    parser.add_argument("--out")
    parser.add_argument(
        "--check", action="store_true", help="check the data and exit; trains nothing"
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-patience", type=int, default=4)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--threads", type=int, default=2, help="CPU threads; low, so it can share the machine"
    )
    args = parser.parse_args()
    if args.check:
        train, val = load(args.data, "train"), load(args.data, "validation")
        limit = torque_limits(train["torque"])
        for name, d in (("train", train), ("validation", val)):
            print(
                f"{name}: {d['runs']} runs, {len(d['torque'])} reach rows, per case "
                f"{np.bincount(d['case'], minlength=5).tolist()}; "
                f"reach s per dot median {np.median(d['reach_s']):.1f}, 99% "
                f"{np.percentile(d['reach_s'], 99):.1f}, max {d['reach_s'].max():.1f}; "
                f"finite {bool(np.isfinite(d['x']).all() and np.isfinite(d['torque']).all())}; "
                f"share of rows beyond limits {100 * np.mean(np.abs(d['torque']) > limit):.3f}%"
            )
        print(
            "largest expert torque per joint, N m:",
            np.round(np.abs(train["torque"]).max(axis=0), 2).tolist(),
        )
        print("torque limits per joint, N m:        ", np.round(limit, 2).tolist())
        print(
            "expert torque minus gravity, RMSE:   ",
            np.round(
                np.sqrt(np.mean((train["torque"] - train["gravity"]) ** 2, axis=0)), 2
            ).tolist(),
        )
        print(
            "expert torque alone, RMSE:           ",
            np.round(np.sqrt(np.mean(train["torque"] ** 2, axis=0)), 2).tolist(),
        )
        print("Check passed. Nothing was trained.")
        return
    if not args.out:
        parser.error("--out is required unless you use --check")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    fit(load(args.data, "train"), load(args.data, "validation"), out, args)


if __name__ == "__main__":
    main()
