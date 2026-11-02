"""Train the MLP by imitating the expert: lidar history, own pose and velocity, all remaining dots -> 4 wheel speeds.

    python train_mlp.py --check                      check the dataset only; trains nothing
    python train_mlp.py --out runs/mlp1              train; saves best.pt, metrics.csv, curves.png, result.json
    python train_mlp.py --out runs/mlp1 --resume     continue an interrupted training

The test split is never read here. Inputs (199): 5 lidar frames over the last second, 180; x, y, sin yaw, cos yaw, 4;
forward speed, sideways speed, yaw rate, 3; three dot slots in fixed scenario order, each forward, left, up seen from
the robot plus a 1/0 "still there" flag, 12. The expert's current dot is deliberately not an input.
"""
import argparse
import csv
import glob
import json
import time
from dataclasses import dataclass
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

CASES = ("case1", "case2", "case3", "case4", "case5")
MAX_DOTS = 3
RATE_HZ = 100                                 # the dataset keeps every 10th 1 ms step
HISTORY_S = (0.0, 0.25, 0.5, 0.75, 1.0)       # lidar frames, seconds before now
OFFSETS = np.array([int(round(s*RATE_HZ)) for s in HISTORY_S])
STOPPED_RAD_S = 0.05                          # expert wheels below this count as "stopped"
WINDOW = 10                                   # epochs compared when judging convergence
FLAT, RISING = 0.01, 0.05                     # val loss: under 1% better is flat; 5% above its best is rising
COLUMNS = ["epoch", "train_loss", "train_check_loss", "val_loss", "val_r2", "val_rmse_rad_s", "val_rmse_moving", "val_rmse_stopped",
           *[f"val_rmse_{c}" for c in CASES], "learning_rate", "seconds"]
LIDAR_NAMES = [f"lidar_t-{s}s_{10*k}deg" for s in HISTORY_S for k in range(36)]
STATE_NAMES = ["x", "y", "sin_yaw", "cos_yaw", "v_forward", "v_left", "yaw_rate"] + \
              [f"dot{d}_{n}" for d in range(MAX_DOTS) for n in ("forward", "left", "up", "there")]


@dataclass
class Data:
    lidar: np.ndarray    # (rows, 36)
    state: np.ndarray    # (rows, 19)
    wheels: np.ndarray   # (rows, 4)
    start: np.ndarray    # (rows,): row where this row's run begins, so history never crosses runs
    case: np.ndarray     # (rows,): index into CASES
    runs: int


def state_rows(obs, t, targets, picks):
    """Inputs other than lidar, for every row of one run. Returns (rows, 19) and the "still there" flags."""
    c, s = obs[:, 4], obs[:, 3]
    vx, vy = obs[:, 22], obs[:, 23]
    out = np.zeros((len(obs), len(STATE_NAMES)), np.float32)
    out[:, :4] = obs[:, [0, 1, 3, 4]]
    out[:, 4], out[:, 5], out[:, 6] = c*vx + s*vy, -s*vx + c*vy, obs[:, 27]
    there = np.zeros((len(obs), MAX_DOTS), bool)
    for d, target in enumerate(targets):
        gone_at = picks.get(d, np.inf)
        there[:, d] = t < gone_at
        dx, dy = target[0] - obs[:, 0], target[1] - obs[:, 1]
        slot = np.stack([c*dx + s*dy, c*dy - s*dx, target[2] - obs[:, 2], np.ones(len(obs))], axis=1)
        out[:, 7 + 4*d:11 + 4*d] = slot*there[:, d, None]        # a removed or unused slot is all zeros
    return out, there


def load(folder, split, limit=None, dagger=()):
    """Read one split of the clean expert runs, plus (for training) every run in the DAgger folders, where the MLP drove
    and the label is what the expert would have done. Fails loudly if the expert's current dot was ever marked as gone."""
    lidar, state, wheels, start, case = [], [], [], [], []
    runs, rows = 0, 0
    files = sorted(glob.glob(f"{folder}/case*/part*/*/trajectory.npz"))
    expert_files = len(files)
    for extra in dagger:
        files += sorted(glob.glob(f"{extra}/*/trajectory.npz"))
    for number, name in enumerate(files):
        learned = number >= expert_files
        folder_ = Path(name).parent
        summary = json.loads((folder_/"summary.json").read_text())
        scenario = json.loads((folder_/"episode.json").read_text())["scenario"]
        if not learned and (scenario["split"] != split or not summary["success"] or (limit and runs >= limit)):
            continue
        z = np.load(name)
        obs, t = z["observations"], z["time_s"]
        targets = np.array(scenario.get("targets", [scenario["target"]]))
        rate = round(1/np.median(np.diff(t)))
        if rate != RATE_HZ or len(targets) > MAX_DOTS or (not learned and summary["dots_picked"] != len(targets)):
            raise ValueError(f"{folder_}: {rate} Hz, {len(targets)} dots, {summary['dots_picked']} picked")
        rows_state, there = state_rows(obs, t, targets, {p["dot"]: p["time_s"] for p in summary["picks"]})
        current = np.abs(obs[:, 5:8, None] - targets.T[None]).max(axis=1).argmin(axis=1)
        if not there[np.arange(len(t)), current].all():   # holds in DAgger runs too: the current dot is always a remaining one
            raise ValueError(f"{folder_}: the dot the expert is going to is marked as gone")
        lidar.append(z["lidar"].astype(np.float32))
        state.append(rows_state)
        wheels.append(z["expert_wheels" if learned else "actions"][:, :4].astype(np.float32))
        start.append(np.full(len(t), rows, np.int64))
        case.append(np.full(len(t), CASES.index(scenario["case"]), np.int8))
        rows += len(t)
        runs += 1
    if not runs:
        raise SystemExit(f"No {split} runs found in {folder}")
    parts = []
    for rows_list in (lidar, state, wheels, start, case):     # free each list as it is joined, so memory does not double
        parts.append(np.concatenate(rows_list))
        rows_list.clear()
    return Data(*parts, runs)


class Scale:
    """Input and output scaling, learned from the training rows only."""
    def __init__(self, train=None, saved=None):
        if saved is not None:
            self.x_mean, self.x_std, self.y_mean, self.y_std = (saved[k] for k in ("x_mean", "x_std", "y_mean", "y_std"))
            return
        self.x_mean = np.concatenate([np.tile(train.lidar.mean(axis=0), len(OFFSETS)), train.state.mean(axis=0)]).astype(np.float32)
        self.x_std = np.concatenate([np.tile(train.lidar.std(axis=0), len(OFFSETS)), train.state.std(axis=0)]).astype(np.float32) + 1e-6
        self.y_mean, self.y_std = train.wheels.mean(axis=0), train.wheels.std(axis=0) + 1e-6

    def inputs(self, data, idx):
        past = np.maximum(idx[:, None] - OFFSETS, data.start[idx, None])     # at a run's start, repeat its first frame
        x = np.hstack([data.lidar[past].reshape(len(idx), -1), data.state[idx]])
        return torch.from_numpy((x - self.x_mean)/self.x_std)

    def outputs(self, data, idx):
        return torch.from_numpy((data.wheels[idx] - self.y_mean)/self.y_std)


def make_model(hidden, layers):
    parts, size = [], len(LIDAR_NAMES) + len(STATE_NAMES)
    for _ in range(layers):
        parts += [nn.Linear(size, hidden), nn.ReLU()]
        size = hidden
    return nn.Sequential(*parts, nn.Linear(size, 4))


def load_policy(path):
    """Rebuild a saved model for driving."""
    saved = torch.load(path, map_location="cpu", weights_only=False)
    model = make_model(saved["hidden"], saved["layers"])
    model.load_state_dict(saved["state_dict"])
    return model.eval(), saved


@torch.no_grad()
def predict(model, scale, data, idx, batch=8192):
    model.eval()
    out = [model(scale.inputs(data, idx[i:i + batch])).numpy() for i in range(0, len(idx), batch)]
    return np.concatenate(out)*scale.y_std + scale.y_mean


def scaled_loss(pred, truth, scale):
    """Mean squared wheel error in scaled units: 1.0 means no better than always predicting the average command."""
    return float(np.mean(((pred - truth)/scale.y_std)**2))


def rmse(pred, truth, mask=None):
    if mask is not None:
        if not mask.any():
            return float("nan")
        pred, truth = pred[mask], truth[mask]
    return float(np.sqrt(np.mean((pred - truth)**2)))


def verdict(rows):
    """Judge convergence from the per-epoch metrics, using the validation runs. Returns (label, explanation)."""
    val, check = [r["val_loss"] for r in rows], [r["train_check_loss"] for r in rows]
    if len(val) <= WINDOW:
        return "TOO SHORT", f"only {len(val)} epochs; need more than {WINDOW}"
    best = int(np.argmin(val))
    if val[-1] > (1 + RISING)*val[best] and check[-1] < check[best]:
        return "OVERFITTING", (f"val loss rose from {val[best]:.4f} at epoch {best + 1} to {val[-1]:.4f} while train kept falling; "
                               f"best.pt keeps epoch {best + 1}")
    before, recent = min(val[:-WINDOW]), min(val[-WINDOW:])
    gain = (before - recent)/before
    if gain >= FLAT:
        return "NOT CONVERGED", f"val loss still improved {100*gain:.1f}% in the last {WINDOW} epochs; train longer with --resume --epochs"
    label = "CONVERGED" if val[best] < 0.5 else "CONVERGED BUT WEAK"
    return label, (f"val loss improved only {100*max(gain, 0):.1f}% in the last {WINDOW} epochs; best {val[best]:.4f} at epoch {best + 1}, "
                   f"R2 {1 - val[best]:.3f}")


def plot(rows, path):
    epochs = [r["epoch"] for r in rows]
    figure, (a, b) = plt.subplots(1, 2, figsize=(11, 4))
    a.plot(epochs, [r["train_check_loss"] for r in rows], label="train")
    a.plot(epochs, [r["val_loss"] for r in rows], label="validation")
    a.set(xlabel="epoch", ylabel="scaled MSE (1 = predicts the average)", yscale="log", title="Loss")
    a.legend()
    for c in CASES:
        b.plot(epochs, [r[f"val_rmse_{c}"] for r in rows], label=c)
    b.set(xlabel="epoch", ylabel="wheel RMSE (rad/s)", title="Validation error per case")
    b.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def fit(train, val, out, args):
    """Training loop. Every epoch appends to metrics.csv and redraws curves.png, so progress can be read at any time."""
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    scale = Scale(train)
    model = make_model(args.hidden, args.layers)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, factor=0.5, patience=args.lr_patience, threshold=0.001)
    loss_fn = nn.MSELoss()
    rows, best_loss, stale = [], float("inf"), 0
    if args.resume:
        saved = torch.load(out/"last.pt", weights_only=False)
        model.load_state_dict(saved["model"])
        optimiser.load_state_dict(saved["optimiser"])
        scheduler.load_state_dict(saved["scheduler"])
        rows, best_loss, stale = saved["rows"], saved["best_loss"], saved["stale"]
    # A fixed set of training rows, as many as the validation rows, measured the same way as validation
    check = rng.choice(len(train.wheels), min(len(val.wheels), len(train.wheels)), replace=False)
    all_val = np.arange(len(val.wheels))
    moving = np.abs(val.wheels).max(axis=1) >= STOPPED_RAD_S
    settings = {k: v for k, v in vars(args).items()}
    print(f"train {train.runs} runs, {len(train.wheels)} rows; validation {val.runs} runs, {len(val.wheels)} rows; "
          f"{sum(p.numel() for p in model.parameters())} parameters; {torch.get_num_threads()} threads", flush=True)
    for epoch in range(len(rows) + 1, args.epochs + 1):
        begin = time.time()
        model.train()
        total, order = 0.0, torch.randperm(len(train.wheels)).numpy()
        for i in range(0, len(order), args.batch):
            idx = order[i:i + args.batch]
            loss = loss_fn(model(scale.inputs(train, idx)), scale.outputs(train, idx))
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            total += loss.item()*len(idx)
        pred = predict(model, scale, val, all_val)
        val_loss = scaled_loss(pred, val.wheels, scale)
        if not np.isfinite(val_loss):
            raise SystemExit(f"Stopped: the validation loss became {val_loss} at epoch {epoch}")
        row = {"epoch": epoch, "train_loss": total/len(order),
               "train_check_loss": scaled_loss(predict(model, scale, train, check), train.wheels[check], scale),
               "val_loss": val_loss, "val_r2": 1 - val_loss, "val_rmse_rad_s": rmse(pred, val.wheels),
               "val_rmse_moving": rmse(pred, val.wheels, moving), "val_rmse_stopped": rmse(pred, val.wheels, ~moving),
               **{f"val_rmse_{c}": rmse(pred, val.wheels, val.case == k) for k, c in enumerate(CASES)},
               "learning_rate": optimiser.param_groups[0]["lr"], "seconds": time.time() - begin}
        rows.append(row)
        scheduler.step(val_loss)
        improved = val_loss < (1 - 0.001)*best_loss
        stale = 0 if improved else stale + 1
        if improved:
            best_loss = val_loss
            torch.save({"state_dict": model.state_dict(), "hidden": args.hidden, "layers": args.layers, "epoch": epoch,
                        "input_names": LIDAR_NAMES + STATE_NAMES, "history_s": HISTORY_S, "x_mean": scale.x_mean,
                        "x_std": scale.x_std, "y_mean": scale.y_mean, "y_std": scale.y_std}, out/"best.pt")
        torch.save({"model": model.state_dict(), "optimiser": optimiser.state_dict(), "scheduler": scheduler.state_dict(),
                    "rows": rows, "best_loss": best_loss, "stale": stale}, out/"last.pt")
        with (out/"metrics.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        plot(rows, out/"curves.png")
        print(f"epoch {epoch:3d}  train {row['train_check_loss']:.4f}  val {val_loss:.4f}  R2 {row['val_r2']:.3f}  "
              f"RMSE {row['val_rmse_rad_s']:.3f} rad/s (moving {row['val_rmse_moving']:.3f}, stopped {row['val_rmse_stopped']:.3f})  "
              f"lr {row['learning_rate']:.1e}  {row['seconds']:.0f} s{'  *best' if improved else ''}", flush=True)
        if stale >= args.patience:
            print(f"Early stop: no improvement for {stale} epochs", flush=True)
            break
    label, why = verdict(rows)
    best = min(rows, key=lambda r: r["val_loss"])
    (out/"result.json").write_text(json.dumps({"verdict": label, "why": why, "best": best, "epochs_run": len(rows),
                                               "train_runs": train.runs, "val_runs": val.runs, "settings": settings}, indent=1))
    print(f"\nVERDICT: {label}\n{why}")
    print("Best epoch per case, RMSE rad/s: " + ", ".join(f"{c} {best[f'val_rmse_{c}']:.3f}" for c in CASES))
    print(f"Saved {out/'best.pt'}. Good wheel numbers do not yet prove it can drive: that takes the closed-loop test.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="runs/dataset")
    parser.add_argument("--out")
    parser.add_argument("--dagger", nargs="*", default=[], help="folders of DAgger runs to add to the training rows")
    parser.add_argument("--check", action="store_true", help="check the dataset and exit; trains nothing")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-patience", type=int, default=4, help="halve the learning rate after this many flat epochs")
    parser.add_argument("--patience", type=int, default=12, help="stop after this many epochs without improvement")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.check:
        for split in ("train", "validation"):
            d = load(args.data, split)
            print(f"{split}: {d.runs} runs, {len(d.wheels)} rows, per case {np.bincount(d.case, minlength=5).tolist()}, "
                  f"stopped rows {100*np.mean(np.abs(d.wheels).max(axis=1) < STOPPED_RAD_S):.1f}%, "
                  f"finite {bool(np.isfinite(d.lidar).all() and np.isfinite(d.state).all() and np.isfinite(d.wheels).all())}, "
                  f"memory {(d.lidar.nbytes + d.state.nbytes + d.wheels.nbytes)/1e9:.2f} GB")
        print("Dataset check passed. Nothing was trained.")
        return
    if not args.out:
        parser.error("--out is required unless you use --check")
    out = Path(args.out)
    if args.resume != out.exists():
        raise SystemExit(f"{out}: use a new folder, or --resume with an existing one")
    out.mkdir(parents=True, exist_ok=True)
    fit(load(args.data, "train", dagger=args.dagger), load(args.data, "validation"), out, args)


if __name__ == "__main__":
    main()
