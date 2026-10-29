"""Train the lidar base policy in PyTorch: 9 base features + 36 lidar ranges -> 4 wheel speeds.

Run with no options to train exactly as before (runs/obst-v2b -> runs/obst-base-torch-v1). Options:
    --data FOLDER ...     expert datasets; label = the expert's own wheel commands ("actions")
    --dagger FOLDER ...   DAgger datasets (learned policy drove); label = "expert_wheels", what the expert
                          would have done in each visited state
    --no-velocity         drop the 3 velocity inputs (v_forward, v_left, yaw_rate) -> 42 inputs
    --balance             weighted loss: rows from hard layouts (scenario id contains "hard") and normal rows
                          count equally in total, however many rows each group has
    --out FOLDER          where to save model.pt
The saved weights are those of the epoch with the lowest validation loss (early stopping), not the last one.

Example:
    python train_base_torch.py --data runs/obst-v2b runs/obst-v3-extra-0 --no-velocity --out runs/my-model
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from models import base_inputs, BASE_INPUT_NAMES

ROOT = Path(__file__).resolve().parent
DATA = ROOT/"runs"/"obst-v2b"
OUT = ROOT/"runs"/"obst-base-torch-v1"
STRIDE = 10            # keep every 10th row: neighbouring 1 ms rows are near-copies
HIDDEN = 64
EPOCHS = 30
BATCH = 256
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INPUT_NAMES = BASE_INPUT_NAMES + [f"lidar_{10*k}deg_m" for k in range(36)]
VELOCITY = ("v_forward_m_s", "v_left_m_s", "yaw_rate_rad_s")


def load(split, folder=DATA, label="actions"):
    """All usable episodes of one split in one dataset folder -> inputs (rows, 45), wheel targets (rows, 4) and
    a hard-layout flag per row (rows,).

    label "actions": an expert dataset; only successful episodes, target = the expert's own wheel commands.
    label "expert_wheels": a DAgger dataset; every episode (the learned policy may have failed, that is the
    point), target = what the expert would have done in each state the policy visited.
    """
    folder = Path(folder)
    manifest = json.loads((folder/"manifest.json").read_text())
    stride = max(1, STRIDE // manifest.get("saved_every", 1))   # DAgger runs are already saved every 10th step
    xs, ys, hard = [], [], []
    for ep in manifest["episodes"]:
        if ep["split"] != split or (label == "actions" and not ep["success"]):
            continue
        z = np.load(folder/ep["path"]/"trajectory.npz")
        obs = z["observations"][::stride]
        # DAgger runs with noisy sensors: train on what the policy SAW, labelled by what the expert would do
        lidar = z["lidar_seen" if label == "expert_wheels" and "lidar_seen" in z else "lidar"][::stride]
        wheels = z["actions"][::stride, :4] if label == "actions" else z["expert_wheels"][::stride]
        base = np.array([base_inputs(row) for row in obs])      # 9 base features for every row
        xs.append(np.hstack([base, lidar]))                      # + 36 lidar ranges -> (rows, 45)
        ys.append(wheels)
        hard.append(np.full(len(wheels), "hard" in ep.get("scenario_id", ep["path"])))
    if not xs:
        return np.zeros((0, len(INPUT_NAMES)), np.float32), np.zeros((0, 4), np.float32), np.zeros(0, bool)
    return np.concatenate(xs).astype(np.float32), np.concatenate(ys).astype(np.float32), np.concatenate(hard)


def noisy_copy(x, rng, noise_max, dropout_max):
    """Copy of input rows with lidar noise and dropouts (each row gets its own random strength up to the max).
    The labels stay the expert's clean-sensor actions: "this is what you might see; this is still the right move"."""
    x = x.copy()
    lid = slice(len(BASE_INPUT_NAMES), len(INPUT_NAMES))
    n = len(x)
    sigma = rng.uniform(0, noise_max, (n, 1))
    drop = rng.random((n, 36)) < rng.uniform(0, dropout_max, (n, 1))
    noisy = x[:, lid] + sigma*rng.standard_normal((n, 36))
    noisy[drop] = 4.0
    x[:, lid] = np.clip(noisy, 0.5, 4.0)
    return x


def train(data=(DATA,), dagger=(), out=OUT, no_velocity=False, epochs=EPOCHS, seed=0, validation=DATA,
          lidar_noise=0.0, lidar_dropout=0.0, balance=False):
    """Train one model and save it to out/model.pt. Returns the validation wheel RMSE (rad/s).
    lidar_noise / lidar_dropout > 0: add a noisy copy of the EXPERT rows (noise-robust training).
    balance: weight the loss so hard-layout rows and normal rows count equally in total."""
    out = Path(out)
    if out.exists():
        raise FileExistsError(f"Choose a new output folder: {out}")
    torch.manual_seed(seed)
    parts = [load("train", f) for f in data] + [load("train", f, "expert_wheels") for f in dagger]
    x, y = np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])
    hard = np.concatenate([p[2] for p in parts])
    if lidar_noise > 0 or lidar_dropout > 0:
        n_expert = sum(len(p[0]) for p in parts[:len(data)])
        x = np.concatenate([x, noisy_copy(x[:n_expert], np.random.default_rng(seed), lidar_noise, lidar_dropout)])
        y = np.concatenate([y, y[:n_expert]])
        hard = np.concatenate([hard, hard[:n_expert]])
    # Weighted loss: each group gets half the total weight; the mean weight stays 1
    weights = np.ones(len(x), np.float32)
    if balance and 0 < hard.sum() < len(hard):
        weights[hard] = len(x) / (2*hard.sum())
        weights[~hard] = len(x) / (2*(~hard).sum())
    xv, yv, _ = load("validation", validation)
    columns = [i for i, n in enumerate(INPUT_NAMES) if not (no_velocity and n in VELOCITY)]
    x, xv = x[:, columns], xv[:, columns]
    print(f"train rows {len(x)} ({len(data)} expert + {len(dagger)} DAgger folders, {int(hard.sum())} from hard "
          f"layouts), validation rows {len(xv)}, inputs {x.shape[1]}, device {DEVICE}")

    # Scaling (what StandardScaler did for you in sklearn): learn it on TRAIN only
    x_mean, x_std = x.mean(axis=0), x.std(axis=0) + 1e-6
    y_mean, y_std = y.mean(axis=0), y.std(axis=0) + 1e-6

    def to_tensor(a):
        return torch.tensor(a, device=DEVICE)
    X, Y, W = to_tensor((x - x_mean)/x_std), to_tensor((y - y_mean)/y_std), to_tensor(weights)
    XV, YV = to_tensor((xv - x_mean)/x_std), to_tensor((yv - y_mean)/y_std)

    model = nn.Sequential(
        nn.Linear(x.shape[1], HIDDEN),
        nn.ReLU(),
        nn.Linear(HIDDEN, HIDDEN),
        nn.ReLU(),
        nn.Linear(HIDDEN, 4)
    ).to(DEVICE)
    optimiser = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    best_loss, best_epoch, best_state = float("inf"), 0, None
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(X), device=DEVICE)        # shuffle every epoch
        total = 0.0
        for start in range(0, len(X), BATCH):
            idx = order[start:start+BATCH]
            prediction = model(X[idx])
            if balance:   # weighted mean squared error: each row's error counts by its group weight
                loss = (((prediction - Y[idx])**2).mean(dim=1) * W[idx]).mean()
            else:
                loss = loss_fn(prediction, Y[idx])
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            total += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(XV), YV).item()
        if val_loss < best_loss:     # early stopping: remember the best epoch's weights
            best_loss, best_epoch = val_loss, epoch + 1
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        print(f"epoch {epoch+1:2d}  train loss {total/len(X):.4f}  validation loss {val_loss:.4f}")
    model.load_state_dict(best_state)

    # Validation RMSE in real units (rad/s)
    with torch.no_grad():
        pred = model(XV).cpu().numpy() * y_std + y_mean
    rmse = float(np.sqrt(np.mean((pred - yv)**2)))
    print(f"validation wheel RMSE: {rmse:.3f} rad/s (best epoch {best_epoch}), "
          f"parameters {sum(p.numel() for p in model.parameters())}")

    out.mkdir(parents=True)
    torch.save({"state_dict": model.state_dict(), "hidden": HIDDEN,
                "input_names": [INPUT_NAMES[i] for i in columns], "columns": columns,
                "x_mean": x_mean, "x_std": x_std, "y_mean": y_mean, "y_std": y_std}, out/"model.pt")
    (out/"training.json").write_text(json.dumps({"validation_wheel_rmse_rad_s": rmse, "best_epoch": best_epoch,
        "train_rows": len(x), "epochs": epochs, "stride": STRIDE, "no_velocity": no_velocity, "seed": seed,
        "lidar_noise": lidar_noise, "lidar_dropout": lidar_dropout, "balance": balance,
        "hard_rows": int(hard.sum()), "data": [str(f) for f in data], "dagger": [str(f) for f in dagger], "validation": str(validation)},
        indent=1))
    print(f"saved {out/'model.pt'}")
    return rmse


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", nargs="+", default=[str(DATA)])
    parser.add_argument("--dagger", nargs="*", default=[])
    parser.add_argument("--no-velocity", action="store_true")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lidar-noise", type=float, default=0.0, help="max noise (m) for a noisy copy of expert rows")
    parser.add_argument("--lidar-dropout", type=float, default=0.0, help="max dropout chance for that copy")
    parser.add_argument("--balance", action="store_true", help="hard-layout and normal rows count equally")
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    train(args.data, args.dagger, args.out, args.no_velocity, args.epochs, args.seed,
          lidar_noise=args.lidar_noise, lidar_dropout=args.lidar_dropout, balance=args.balance)


if __name__ == "__main__":
    main()
