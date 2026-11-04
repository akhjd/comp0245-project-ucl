"""Train the MLP by imitating the expert: lidar history, own pose and velocity, all remaining dots -> 4 wheel speeds.

    python train_mlp.py --check                      check the dataset only; trains nothing
    python train_mlp.py --out runs/mlp1              train; saves best.pt, metrics.csv, curves.png, result.json
    python train_mlp.py --out runs/mlp1 --resume     continue an interrupted training
    python train_mlp.py --out runs/mlp3 --no-velocity                 leave forward speed, sideways speed and yaw rate out
    python train_mlp.py --out runs/mlp3-case1 --no-velocity --cases case1
    python train_mlp.py --out runs/mlp2 --mirror --lidar-noise 0.05 --velocity-noise 0.03 --yaw-rate-noise 0.1
    python train_mlp.py --out runs/mlp4 --no-velocity --no-xy         also available: --dot-polar, --lidar-features
    python train_mlp.py --out runs/size-half --no-velocity --dot-polar --run-fraction 0.5      train on half the expert runs
    python train_mlp.py --out runs/reg-both --no-velocity --dot-polar --dropout 0.2 --weight-decay 0.01 --evals-per-epoch 4
    python train_mlp.py --out runs/pre1 --preprocess --no-velocity --dot-polar --dropout 0.2 --weight-decay 0.01 --lr 3e-4 --evals-per-epoch 4
    python train_mlp.py --out runs/pre2 --preprocess --door-memory --no-velocity --dot-polar --dropout 0.2 --weight-decay 0.01 --lr 3e-4 --evals-per-epoch 4

The test split is never read here. Inputs (199): 5 lidar frames over the last second, 180; x, y, sin yaw, cos yaw, 4;
forward speed, sideways speed, yaw rate, 3; three dot slots in fixed scenario order, each forward, left, up seen from
the robot plus a 1/0 "still there" flag, 12. The expert's current dot is deliberately not an input.
"""
import argparse
import csv
import glob
import json
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import preprocess

CASES = ("case1", "case2", "case3", "case4", "case5")
MAX_DOTS = 3
RATE_HZ = 100                                 # the dataset keeps every 10th 1 ms step
HISTORY_S = (0.0, 0.25, 0.5, 0.75, 1.0)       # lidar frames, seconds before now
OFFSETS = np.array([int(round(s*RATE_HZ)) for s in HISTORY_S])
STOPPED_RAD_S = 0.05                          # expert wheels below this count as "stopped"
WINDOW = 10                                   # epochs compared when judging convergence
FLAT, RISING = 0.01, 0.05                     # val loss: under 1% better is flat; 5% above its best is rising
GAP = 1.25                                    # val must also be this many times the train loss to count as overfitting
COLUMNS = ["epoch", "train_loss", "train_check_loss", "val_loss", "val_r2", "val_rmse_rad_s", "val_rmse_moving", "val_rmse_stopped",
           *[f"val_rmse_{c}" for c in CASES], "learning_rate", "seconds"]
LIDAR_MIN, LIDAR_MAX = 0.5, 4.0               # m: what the lidar can report
# Left-right mirror: ray k looks at +10k degrees, so its twin is ray -k; y, sin yaw, sideways speed, yaw rate and each dot's
# "left" change sign; wheels are ordered front right, front left, back left, back right, so left and right swap.
MIRROR_RAYS = -np.arange(36) % 36
MIRROR_STATE = np.array([1, -1, -1, 1, 1, -1, -1, *[1, -1, 1, 1]*MAX_DOTS], np.float32)
MIRROR_WHEELS = [1, 0, 3, 2]
LIDAR_NAMES = [f"lidar_t-{s}s_{10*k}deg" for s in HISTORY_S for k in range(36)]
VELOCITY_NAMES = ("v_forward", "v_left", "yaw_rate")
STATE_NAMES = ["x", "y", "sin_yaw", "cos_yaw", "v_forward", "v_left", "yaw_rate"] + \
              [f"dot{d}_{n}" for d in range(MAX_DOTS) for n in ("forward", "left", "up", "there")]
DOT_POLAR_NAMES = [f"dot{d}_{n}" for d in range(MAX_DOTS) for n in ("distance", "sin_heading", "cos_heading", "up", "there")]
LIDAR_FEATURE_NAMES = [f"sector{10*3*j}deg_nearest" for j in range(12)] + [f"sector{10*3*j}deg_change" for j in range(12)] + \
                      ["closest_distance", "closest_sin", "closest_cos"] + [f"dot{d}_free" for d in range(MAX_DOTS)]
NO_INPUT_CHANGES = {"no_velocity": False, "no_xy": False, "dot_polar": False, "lidar_features": False, "preprocess": False,
                    "door_memory": False}


@dataclass
class Data:
    lidar: np.ndarray    # (rows, 36)
    state: np.ndarray    # (rows, 19)
    wheels: np.ndarray   # (rows, 4)
    start: np.ndarray    # (rows,): row where this row's run begins, so history never crosses runs
    case: np.ndarray     # (rows,): index into CASES
    runs: int
    pre: np.ndarray = None   # (rows, len(preprocess.MLP_NAMES)) with --preprocess; lidar is then left empty


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


def pre_version(door_memory=False):
    """Which preprocessed feature set (and cache file): v1, or v2 with the remembered doorways."""
    return "v2" if door_memory else "v1"


def feature_names(dot_polar=False, lidar_features=False, preprocess_=False, door_memory=False):
    """Names of the inputs after make_features, before any are left out."""
    lidar = preprocess.FEATURE_SETS[pre_version(door_memory)] if preprocess_ else LIDAR_FEATURE_NAMES if lidar_features else LIDAR_NAMES
    return lidar + STATE_NAMES[:7] + (DOT_POLAR_NAMES if dot_polar else STATE_NAMES[7:])


def kept_columns(no_velocity=False, no_xy=False, dot_polar=False, lidar_features=False, preprocess=False, door_memory=False):
    """Which of the inputs the network gets."""
    return np.array([i for i, n in enumerate(feature_names(dot_polar, lidar_features, preprocess, door_memory))
                     if not (no_velocity and n in VELOCITY_NAMES) and not (no_xy and n in ("x", "y"))])


def make_features(x, dot_polar=False, lidar_features=False, preprocess_=False, door_memory=False):
    """Turn the 199 raw inputs into the chosen features. Training and driving both use this, so they cannot disagree.
    dot_polar: each dot's forward and left become distance and sin, cos of its heading from the robot (5 values per slot).
    lidar_features: 180 rays become 30 values: nearest obstacle per 30 degree sector now, its change over 0.5 s, the closest
    obstacle's distance and bearing, and the free distance towards each dot. A removed or unused slot is all zeros, except
    its free distance, which reads as clear.
    preprocess_: x is the preprocessed lidar features (preprocess.FEATURE_SETS, v2 with door_memory) followed by the 19
    state values; only the dots can still change."""
    if not (dot_polar or lidar_features):
        return x
    n, width = len(x), len(preprocess.FEATURE_SETS[pre_version(door_memory)]) if preprocess_ else len(LIDAR_NAMES)
    state = x[:, width:]
    forward, left, up, there = np.moveaxis(state[:, 7:].reshape(n, MAX_DOTS, 4), 2, 0)
    heading = np.arctan2(left, forward)
    if lidar_features:
        frames = x[:, :width].reshape(n, len(OFFSETS), 36)
        now, before = frames[:, 0], frames[:, list(HISTORY_S).index(0.5)]
        sectors = lambda f: np.roll(f, 1, axis=1).reshape(n, 12, 3).min(axis=2)     # sector j covers rays 3j-1, 3j, 3j+1
        nearest = sectors(now)
        closest = np.radians(10.0*now.argmin(axis=1))
        ray = np.rint(np.degrees(heading)/10).astype(int) % 36
        free = np.where(there > 0, now[np.arange(n)[:, None], ray], LIDAR_MAX)
        lidar = np.hstack([nearest, nearest - sectors(before), now.min(axis=1, keepdims=True),
                           np.sin(closest)[:, None], np.cos(closest)[:, None], free])
    else:
        lidar = x[:, :width]
    if dot_polar:
        dots = np.stack([np.hypot(forward, left)*there, np.sin(heading)*there, np.cos(heading)*there, up, there], axis=2).reshape(n, -1)
    else:
        dots = state[:, 7:]
    return np.hstack([lidar, state[:, :7], dots]).astype(np.float32)


def load(folder, split, limit=None, dagger=(), cases=CASES, run_fraction=1.0, preprocess_=False, version="v1"):
    """Read one split of the clean expert runs, plus (for training) every run in the DAgger folders, where the MLP drove
    and the label is what the expert would have done. Fails loudly if the expert's current dot was ever marked as gone.
    preprocess_: keep the preprocessed lidar features (feature set `version`, cached next to each run) instead of the raw
    lidar."""
    lidar, state, wheels, start, case, pre = [], [], [], [], [], []
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
        if scenario["case"] not in cases:
            continue
        if not learned and (scenario["split"] != split or not summary["success"] or (limit and runs >= limit)):
            continue
        # A fixed subset of expert runs, chosen by scenario id, so a smaller fraction is always inside a larger one
        if not learned and run_fraction < 1 and zlib.crc32(scenario["id"].encode())/2**32 >= run_fraction:
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
        if preprocess_:
            pre.append(preprocess.cached_features(folder_, t, z["lidar"], obs, rows_state[:, 7:].reshape(-1, MAX_DOTS, 4), version))
            lidar.append(np.zeros((len(t), 0), np.float32))
        else:
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
    return Data(*parts, runs, np.concatenate(pre) if preprocess_ else None)


class Scale:
    """Input and output scaling, learned from the training rows only."""
    def __init__(self, train=None, saved=None, mirror=False, keep=None, changes=None):
        self.keep = kept_columns(False) if keep is None else keep
        self.changes = {**NO_INPUT_CHANGES, **(changes or {})}
        if saved is not None:
            self.x_mean, self.x_std, self.y_mean, self.y_std = (saved[k] for k in ("x_mean", "x_std", "y_mean", "y_std"))
            self.keep = saved.get("keep", self.keep)                    # older models have no "keep": all 199 inputs
            self.changes = {**NO_INPUT_CHANGES, **saved.get("input_changes", {})}   # nor "input_changes": raw inputs
            return
        def stats(rows, flip):
            """Mean and std of the rows; with mirror, of the rows and their mirrored copies together."""
            mean, variance = rows.mean(axis=0), rows.var(axis=0)
            if mirror:
                other = flip(mean)
                mean, variance = (mean + other)/2, (variance + np.abs(flip(variance)))/2 + ((mean - other)/2)**2
            return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)
        lidar_mean, lidar_std = stats(train.lidar, lambda a: a[MIRROR_RAYS])
        state_mean, state_std = stats(train.state, lambda a: a*MIRROR_STATE)
        self.x_mean = np.concatenate([np.tile(lidar_mean, len(OFFSETS)), state_mean])
        self.x_std = np.concatenate([np.tile(lidar_std, len(OFFSETS)), state_std]) + 1e-6
        self.y_mean, y_std = stats(train.wheels, lambda a: a[MIRROR_WHEELS])
        self.y_std = y_std + 1e-6
        if self.changes["dot_polar"] or self.changes["lidar_features"] or self.changes["preprocess"]:   # scale from a large sample
            sample = np.random.default_rng(0).choice(len(train.wheels), min(200_000, len(train.wheels)), replace=False)
            x = self.features(self.raw(train, sample))
            self.x_mean, self.x_std = x.mean(axis=0), x.std(axis=0) + 1e-6

    def raw(self, data, idx):
        """Rows before scaling: lidar history (or the preprocessed lidar features) followed by the 19 state values."""
        if self.changes["preprocess"]:
            return np.hstack([data.pre[idx], data.state[idx]])
        past = np.maximum(idx[:, None] - OFFSETS, data.start[idx, None])     # at a run's start, repeat its first frame
        return np.hstack([data.lidar[past].reshape(len(idx), -1), data.state[idx]])

    def features(self, x):
        return make_features(x, self.changes["dot_polar"], self.changes["lidar_features"], self.changes["preprocess"],
                             self.changes["door_memory"])

    def prepare(self, x):
        """Raw (rows, 199) inputs to what the network receives."""
        return ((self.features(x) - self.x_mean)/self.x_std)[:, self.keep]

    def inputs(self, data, idx):
        return torch.from_numpy(self.prepare(self.raw(data, idx)))

    def outputs(self, data, idx):
        return torch.from_numpy((data.wheels[idx] - self.y_mean)/self.y_std)

    def training_batch(self, data, idx, args, rng):
        """Scaled inputs and outputs with this batch's noise and mirroring. Each row gets its own noise strength, from none
        up to the maximum. Half the rows are mirrored, chosen afresh every time, so memory use does not grow."""
        n = len(idx)
        if self.changes["preprocess"]:          # no noise or mirroring here (main() refuses them)
            return self.inputs(data, idx), self.outputs(data, idx)
        past = np.maximum(idx[:, None] - OFFSETS, data.start[idx, None])
        lidar, state, wheels = data.lidar[past], data.state[idx].copy(), data.wheels[idx].copy()
        if args.lidar_noise > 0:
            strength = rng.uniform(0, args.lidar_noise, (n, 1, 1))
            lidar = np.clip(lidar + (strength*rng.standard_normal(lidar.shape)).astype(np.float32), LIDAR_MIN, LIDAR_MAX)
        if args.velocity_noise > 0:
            state[:, 4:6] += (rng.uniform(0, args.velocity_noise, (n, 1))*rng.standard_normal((n, 2))).astype(np.float32)
        if args.yaw_rate_noise > 0:
            state[:, 6] += (rng.uniform(0, args.yaw_rate_noise, n)*rng.standard_normal(n)).astype(np.float32)
        if args.mirror:
            flip = rng.random(n) < 0.5
            lidar[flip] = lidar[flip][:, :, MIRROR_RAYS]
            state[flip] *= MIRROR_STATE
            wheels[flip] = wheels[flip][:, MIRROR_WHEELS]
        x = np.hstack([lidar.reshape(n, -1), state])
        return torch.from_numpy(self.prepare(x)), torch.from_numpy((wheels - self.y_mean)/self.y_std)


def make_model(hidden, layers, inputs=None, dropout=0.0):
    parts, size = [], inputs or len(LIDAR_NAMES) + len(STATE_NAMES)
    for _ in range(layers):
        parts += [nn.Linear(size, hidden), nn.ReLU()] + ([nn.Dropout(dropout)] if dropout > 0 else [])
        size = hidden
    return nn.Sequential(*parts, nn.Linear(size, 4))


def load_policy(path):
    """Rebuild a saved model for driving."""
    saved = torch.load(path, map_location="cpu", weights_only=False)
    model = make_model(saved["hidden"], saved["layers"], len(saved["input_names"]), saved.get("training", {}).get("dropout", 0.0))
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
    recent_val, recent_check = np.mean(val[-5:]), np.mean(check[-5:])     # averaged, as one epoch's value is noisy
    if recent_val > (1 + RISING)*val[best] and recent_check < check[best] and recent_val > GAP*recent_check:
        return "OVERFITTING", (f"val loss averaged {recent_val:.4f} over the last 5 epochs, above its best {val[best]:.4f} at epoch {best + 1}, "
                               f"while train kept falling to {recent_check:.4f}; best.pt keeps epoch {best + 1}")
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
    changes = {k: getattr(args, k, False) for k in NO_INPUT_CHANGES}
    keep = kept_columns(**changes)
    names = feature_names(args.dot_polar, args.lidar_features, changes["preprocess"], changes["door_memory"])
    scale = Scale(train, mirror=args.mirror, keep=keep, changes=changes)
    model = make_model(args.hidden, args.layers, len(keep), args.dropout)
    if args.weight_decay > 0:      # AdamW: weight decay applied separately from the gradient step
        optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)
    training = {"dropout": args.dropout, "weight_decay": args.weight_decay, "evals_per_epoch": args.evals_per_epoch, "lr": args.lr}
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
    per = args.evals_per_epoch          # validation checks per epoch; --patience and --lr-patience count checks
    stop = False
    for epoch in range(len(rows)//per + 1, args.epochs + 1):
        order = torch.randperm(len(train.wheels)).numpy()
        for part, batches in enumerate(np.array_split(np.arange(0, len(order), args.batch), per)):
            begin, total, seen = time.time(), 0.0, 0
            model.train()
            for i in batches:
                idx = order[i:i + args.batch]
                inputs, targets = scale.training_batch(train, idx, args, rng)
                loss = loss_fn(model(inputs), targets)
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
                total, seen = total + loss.item()*len(idx), seen + len(idx)
            at = epoch if per == 1 else round(epoch - 1 + (part + 1)/per, 3)
            pred = predict(model, scale, val, all_val)
            val_loss = scaled_loss(pred, val.wheels, scale)
            if not np.isfinite(val_loss):
                raise SystemExit(f"Stopped: the validation loss became {val_loss} at epoch {at}")
            row = {"epoch": at, "train_loss": total/seen,
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
                torch.save({"state_dict": model.state_dict(), "hidden": args.hidden, "layers": args.layers, "epoch": at, "training": training,
                            "input_names": [names[i] for i in keep], "keep": keep, "input_changes": changes, "history_s": HISTORY_S, "x_mean": scale.x_mean,
                            "x_std": scale.x_std, "y_mean": scale.y_mean, "y_std": scale.y_std}, out/"best.pt")
            with (out/"metrics.csv").open("w", newline="") as f:
                writer = csv.DictWriter(f, COLUMNS)
                writer.writeheader()
                writer.writerows(rows)
            plot(rows, out/"curves.png")
            print(f"epoch {at:6.2f}  train {row['train_check_loss']:.4f}  val {val_loss:.4f}  R2 {row['val_r2']:.3f}  "
                  f"RMSE {row['val_rmse_rad_s']:.3f} rad/s (moving {row['val_rmse_moving']:.3f}, stopped {row['val_rmse_stopped']:.3f})  "
                  f"lr {row['learning_rate']:.1e}  {row['seconds']:.0f} s{'  *best' if improved else ''}", flush=True)
            if stale >= args.patience:
                print(f"Early stop: no improvement for {stale} {'epochs' if per == 1 else 'checks'}", flush=True)
                stop = True
                break
        if stop:
            break
        torch.save({"model": model.state_dict(), "optimiser": optimiser.state_dict(), "scheduler": scheduler.state_dict(),   # resume point: end of an epoch
                    "rows": rows, "best_loss": best_loss, "stale": stale}, out/"last.pt")
    label, why = verdict(rows)
    best = min(rows, key=lambda r: r["val_loss"])
    (out/"result.json").write_text(json.dumps({"verdict": label, "why": why, "best": best, "epochs_run": rows[-1]["epoch"], "checks_run": len(rows),
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
    parser.add_argument("--dropout", type=float, default=0.0, help="dropout after each hidden layer, such as 0.2")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="AdamW weight decay, such as 0.01; 0 keeps plain Adam")
    parser.add_argument("--evals-per-epoch", type=int, default=1, help="validate this many times per epoch; patience then counts checks")
    parser.add_argument("--no-velocity", action="store_true", help="leave forward speed, sideways speed and yaw rate out of the inputs")
    parser.add_argument("--run-fraction", type=float, default=1.0, help="train on this fraction of the expert training runs (validation unchanged)")
    parser.add_argument("--no-xy", action="store_true", help="leave the robot's x and y position out of the inputs")
    parser.add_argument("--dot-polar", action="store_true", help="give each dot as distance and heading instead of forward and left")
    parser.add_argument("--lidar-features", action="store_true", help="replace the 180 lidar rays with 30 summary values")
    parser.add_argument("--preprocess", action="store_true", help="replace the lidar history with preprocess.py's features (walls, "
                        "openings, blocked lines, sector changes); cached next to each run")
    parser.add_argument("--door-memory", action="store_true", help="with --preprocess: add the remembered doorways (feature set v2)")
    parser.add_argument("--cases", nargs="+", default=list(CASES), choices=CASES, help="train and validate on these cases only")
    parser.add_argument("--mirror", action="store_true", help="mirror half of each training batch left to right")
    parser.add_argument("--lidar-noise", type=float, default=0.0, help="max noise on training lidar, m")
    parser.add_argument("--velocity-noise", type=float, default=0.0, help="max noise on training forward and sideways speed, m/s")
    parser.add_argument("--yaw-rate-noise", type=float, default=0.0, help="max noise on training yaw rate, rad/s")
    args = parser.parse_args()
    if args.mirror and (args.dot_polar or args.lidar_features):
        parser.error("--mirror has not been checked with --dot-polar or --lidar-features")
    if args.preprocess and (args.mirror or args.lidar_features or args.lidar_noise or args.velocity_noise or args.yaw_rate_noise):
        parser.error("--preprocess works on features, so --mirror, --lidar-features and the noise options do not apply")
    if args.door_memory and not args.preprocess:
        parser.error("--door-memory needs --preprocess")
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
    version = pre_version(args.door_memory)
    fit(load(args.data, "train", dagger=args.dagger, cases=args.cases, run_fraction=args.run_fraction, preprocess_=args.preprocess,
             version=version),
        load(args.data, "validation", cases=args.cases, preprocess_=args.preprocess, version=version), out, args)


if __name__ == "__main__":
    main()
