"""Train the learned arm (model A): 24 arm inputs -> 7 joint torques, on top of the gravity term.
--ik trains model B instead: the same 24 inputs + 7 ik_direction joint speeds
(arm_data.arm_inputs_b) = 31.

    torque = gravity_torque(q) + network(arm_inputs)

Only the reaching phase is used (from the moment the base has parked: arm_data.arm_phase_rows). The
network learns the expert's torque MINUS gravity, so it only has to learn the reaching part.
--dagger FOLDER ...: arm DAgger data (comp0245/training/dagger_arm.py): the learned arm reached,
and the label is what a shadow copy of the expert would have commanded in each state
("expert_arm"). Every DAgger episode is used (failed ones are the point), for training only.
Validation = every 10th episode of each dataset, held out WHOLE: rows next to each other are
near-copies, so splitting row by row would leak the answers into validation. The saved weights are
those of the epoch with the lowest validation loss (early stopping).

Example:
    python -m comp0245.training.train_arm --data runs/obst-v4-height runs/obst-v4-armstart \\
        runs/obst-v4-both --out runs/arm-a
    python -m comp0245.training.train_arm --data runs/obst-v4-height runs/obst-v4-armstart \\
        runs/obst-v4-both --ik --out runs/arm-model-b
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from comp0245.arm_data import ARM_B_INPUT_NAMES, arm_inputs_b, arm_phase_rows, gravity_torque
from comp0245.models import ARM_INPUT_NAMES, arm_inputs

STRIDE = 10  # keep every 10th row: neighbouring 1 ms rows are near-copies
HIDDEN = 64
EPOCHS = 30
BATCH = 256
LR = 1e-3


def load(folders, validation, label="actions", inputs=arm_inputs):
    """Reaching rows -> inputs (rows, 24, or 31 for model B) and torque targets (rows, 7). label
    "actions": expert datasets; successful episodes only; validation=False: the training episodes,
    True:
    every 10th episode (held out). label "expert_arm": DAgger folders; every episode, training
    only."""
    xs, ys = [], []
    for folder in map(Path, folders):
        manifest = json.loads((folder / "manifest.json").read_text())
        stride = max(
            1, STRIDE // manifest.get("saved_every", 1)
        )  # DAgger runs are already saved every 10th step
        for i, ep in enumerate(manifest["episodes"]):
            if label == "actions":
                if not ep["success"] or (i % 10 == 0) != validation:
                    continue
            elif validation:
                continue
            z = np.load(folder / ep["path"] / "trajectory.npz")
            rows = arm_phase_rows(folder, ep)[::stride]
            if len(rows) == 0:
                continue
            obs, ee = z["observations"][rows], z["ee_m"][rows]
            # expert data: the expert's own 7 arm torques (columns 4-10 of the 11 commands);
            # DAgger data: the shadow expert's 7 torques (saved on their own)
            torque = z["actions"][rows, 4:11] if label == "actions" else z["expert_arm"][rows]
            q = obs[:, 8:15]  # the 7 joint angles
            gravity = np.array([gravity_torque(angles) for angles in q])  # (rows, 7)
            x = np.array(
                [inputs(o, e) for o, e in zip(obs, ee, strict=False)]
            )  # arm inputs for every row -> (rows, 24 or 31)
            y = (
                torque - gravity
            )  # TODO 2: torque minus gravity_torque(q) for every row -> (rows, 7)
            xs.append(x)
            ys.append(y)
    return np.concatenate(xs).astype(np.float32), np.concatenate(ys).astype(np.float32)


def train(data, out, epochs=EPOCHS, seed=0, dagger=(), ik=False):
    out = Path(out)
    if out.exists():
        raise FileExistsError(f"Choose a new output folder: {out}")
    torch.manual_seed(seed)
    inputs, names = (arm_inputs_b, ARM_B_INPUT_NAMES) if ik else (arm_inputs, ARM_INPUT_NAMES)
    x, y = load(data, validation=False, inputs=inputs)
    if dagger:
        xd, yd = load(dagger, validation=False, label="expert_arm", inputs=inputs)
        x, y = np.concatenate([x, xd]), np.concatenate([y, yd])
    xv, yv = load(data, validation=True, inputs=inputs)
    print(
        f"train rows {len(x)} ({len(dagger)} DAgger folders), validation rows {len(xv)}, inputs "
        f"{x.shape[1]}"
    )

    # Scaling: learnt on TRAIN only
    x_mean, x_std = x.mean(axis=0), x.std(axis=0) + 1e-6
    y_mean, y_std = y.mean(axis=0), y.std(axis=0) + 1e-6
    X, Y = torch.tensor((x - x_mean) / x_std), torch.tensor((y - y_mean) / y_std)
    XV, YV = torch.tensor((xv - x_mean) / x_std), torch.tensor((yv - y_mean) / y_std)

    model = nn.Sequential(
        nn.Linear(len(names), HIDDEN),
        nn.ReLU(),
        nn.Linear(HIDDEN, HIDDEN),
        nn.ReLU(),
        nn.Linear(HIDDEN, 7),
    )
    optimiser = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    best_loss, best_epoch, best_state = float("inf"), 0, None
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(X))
        total = 0.0
        for start in range(0, len(X), BATCH):
            idx = order[start : start + BATCH]
            loss = loss_fn(model(X[idx]), Y[idx])
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            total += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(XV), YV).item()
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch + 1
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        print(
            f"epoch {epoch + 1:2d}  train loss {total / len(X):.4f}  validation loss "
            f"{val_loss:.4f}"
        )
    model.load_state_dict(best_state)

    # Validation error in real units (N m), per joint, for the network part (the gravity part is
    # exact)
    with torch.no_grad():
        pred = model(XV).numpy() * y_std + y_mean
    rmse = np.sqrt(np.mean((pred - yv) ** 2, axis=0))
    print(f"validation torque RMSE per joint (N m): {np.round(rmse, 2)} (best epoch {best_epoch})")

    out.mkdir(parents=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "hidden": HIDDEN,
            "input_names": names,
            "x_mean": x_mean,
            "x_std": x_std,
            "y_mean": y_mean,
            "y_std": y_std,
        },
        out / "model.pt",
    )
    (out / "training.json").write_text(
        json.dumps(
            {
                "validation_rmse_per_joint_n_m": rmse.round(4).tolist(),
                "best_epoch": best_epoch,
                "train_rows": len(x),
                "validation_rows": len(xv),
                "epochs": epochs,
                "stride": STRIDE,
                "seed": seed,
                "inputs": len(names),
                "data": [str(f) for f in data],
                "dagger": [str(f) for f in dagger],
            },
            indent=1,
        )
    )
    print(f"saved {out / 'model.pt'}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", nargs="+", required=True, help="expert dataset folders")
    parser.add_argument(
        "--dagger",
        nargs="*",
        default=[],
        help="arm DAgger folders (comp0245/training/dagger_arm.py)",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ik", action="store_true", help="model B: add the 7 ik_direction inputs")
    args = parser.parse_args()
    train(args.data, args.out, args.epochs, args.seed, args.dagger, args.ik)


if __name__ == "__main__":
    main()
