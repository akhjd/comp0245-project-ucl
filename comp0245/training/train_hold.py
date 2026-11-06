"""Add a learned "hold" output to a trained base model: the probability that the expert commands
EXACTLY 0 here (parked, or waiting for a mover). LearnedBase
(comp0245/evaluation/eval_obstacles.py) then drives exactly 0 when it is above HOLD_ON.

Why (4 Nov): in the parked state the window expert's DAgger labels are exactly 0, but a regression
output only gets close to 0, so the parked base keeps making tiny turns and fails the "still for
the last 2 s" check (creeping).

The rest of the network is frozen: the hold output is a small head (RayNet.hold) on the model's own
features, trained with cross-entropy on the same training folders as the model (from its
training.json), so the driving is unchanged. The best epoch is picked on the model's validation
folder.

Example:
    python -m comp0245.training.train_hold runs/dagger-rooms-crossing/r3 runs/crossing-hold/r3-hold
"""

import argparse
import ast
import json

import numpy as np
import torch
from torch import nn

from comp0245.evaluation.eval_obstacles import load_model
from comp0245.paths import ROOT
from comp0245.training.train_base_torch import DEVICE, HOLD_HIDDEN, RayNet, load

EPOCHS, BATCH, LR = 8, 4096, 1e-3


def hold_rows(saved, folders, dagger, split):
    """Scaled inputs and the hold label (expert wheels all exactly 0) of the given folders."""
    parts = [load(split, ROOT / f, scans=saved.get("scans", 1)) for f in folders] + [
        load(split, ROOT / f, "expert_wheels", saved.get("scans", 1)) for f in dagger
    ]
    x = np.concatenate([p[0] for p in parts])
    hold = np.concatenate([(p[1] == 0).all(axis=1) for p in parts])
    del parts
    x = x[:, saved["columns"]]
    x -= saved["x_mean"]
    x /= saved["x_std"]
    return x, hold


def features(model, x):
    """The frozen network's features for every row, in chunks (rows, hidden)."""
    with torch.no_grad():
        return torch.cat(
            [
                model.features(torch.as_tensor(x[i : i + 65536]).to(DEVICE)).cpu()
                for i in range(0, len(x), 65536)
            ]
        )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("model", help="folder with model.pt and training.json")
    parser.add_argument("out", help="new folder for the model with the hold output")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    src, out = ROOT / args.model, ROOT / args.out
    info = json.loads((src / "training.json").read_text())
    model, saved = load_model(src / "model.pt")
    if not isinstance(model, RayNet) or model.mixture or model.modes or model.menu:
        raise ValueError("the hold output is for plain regression RayNet models")
    model.to(DEVICE)
    dagger = (
        ast.literal_eval(info["dagger"]) if isinstance(info["dagger"], str) else info["dagger"]
    )  # saved as str(list)

    x, y = hold_rows(saved, info["data"], dagger, "train")
    h = features(model, x)
    del x
    xv, yv = hold_rows(saved, [info["validation"]], [], "validation")
    hv = features(model, xv)
    del xv
    print(
        f"hold rows: train {y.mean():.1%} of {len(y)}, validation {yv.mean():.1%} of {len(yv)}",
        flush=True,
    )

    torch.manual_seed(args.seed)
    head = nn.Sequential(
        nn.Linear(saved["hidden"], HOLD_HIDDEN), nn.ReLU(), nn.Linear(HOLD_HIDDEN, 1)
    ).to(DEVICE)
    opt = torch.optim.Adam(head.parameters(), lr=LR)
    loss_fn = nn.BCEWithLogitsLoss()
    Y, YV = torch.as_tensor(y, dtype=torch.float32), torch.as_tensor(yv, dtype=torch.float32)
    best, best_state, best_epoch, log = np.inf, None, 0, []
    for epoch in range(EPOCHS):
        head.train()
        for idx in torch.randperm(len(h)).split(BATCH):
            opt.zero_grad()
            loss = loss_fn(head(h[idx].to(DEVICE))[:, 0], Y[idx].to(DEVICE))
            loss.backward()
            opt.step()
        head.eval()
        with torch.no_grad():
            logit = torch.cat(
                [head(hv[i : i + 65536].to(DEVICE))[:, 0].cpu() for i in range(0, len(hv), 65536)]
            )
            vloss = float(loss_fn(logit, YV))
        p = (logit > 0).numpy()
        row = {
            "epoch": epoch,
            "validation_bce": vloss,
            "recall": float(p[yv].mean()),
            "false_hold": float(p[~yv].mean()),
        }
        log.append(row)
        print(row, flush=True)
        if vloss < best:
            best, best_epoch = vloss, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}

    model.hold = head.cpu()
    model.hold.load_state_dict(best_state)
    out.mkdir(parents=True)
    torch.save({**saved, "state_dict": model.cpu().state_dict(), "hold": True}, out / "model.pt")
    (out / "training.json").write_text(
        json.dumps(
            {
                **info,
                "hold": True,
                "hold_from": str(args.model),
                "hold_best_epoch": best_epoch,
                "hold_epochs": log,
                "hold_share_train": float(y.mean()),
                "hold_share_validation": float(yv.mean()),
                "hold_seed": args.seed,
            },
            indent=1,
        )
    )
    print(f"saved {out / 'model.pt'} (best epoch {best_epoch})")


if __name__ == "__main__":
    main()
