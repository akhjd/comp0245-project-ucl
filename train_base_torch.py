"""Train the lidar base policy in PyTorch: 9 base features + 36 lidar ranges -> 4 wheel speeds.

The number of lidar rays per scan comes from the datasets' manifest.json ("lidar_rays"; 36 if absent) and is saved
in model.pt as "rays". Every dataset in one training (and the validation set) must have the same ray count.

Run with no options to train exactly as before (runs/obst-v2b -> runs/obst-base-torch-v1). Options:
    --data FOLDER ...     expert datasets; label = the expert's own wheel commands ("actions")
    --dagger FOLDER ...   DAgger datasets (learned policy drove); label = "expert_wheels", what the expert
                          would have done in each visited state
    --no-velocity         drop the 3 velocity inputs (v_forward, v_left, yaw_rate) -> 42 inputs
    --scans N             N lidar scans, SCAN_GAP_S apart (now, 0.5 s ago, ...) -> one more scan of rays per extra scan
    --hidden N            neurons per hidden layer (default 64)
    --encoder conv        learned ray filter (RayNet): one small set of weights looks at 5 neighbouring rays in every
                          scan and slides round all 36 positions (wrapping behind the robot). Its 36 results stay in
                          order, so position = direction. A second filter looks at 5 neighbouring results, then the
                          strongest result per 30 degrees is kept: 36 rays per scan -> 12 sectors x 4 detectors.
                          Other ray counts keep the same angles (filter_sizes): 72 rays -> windows of 9 rays, 6 rays
                          per sector, still 12 sectors x 4 detectors
    --threat              extra training-only output: will a moving obstacle come within THREAT_M of the robot in the
                          next THREAT_S, and where is it and how is it moving (robot's directions). Worked out from
                          the simulator's true positions; never an input, and ignored while driving
    --mixture K           K candidate wheel commands, each with a probability (mixture density network), trained on
                          how likely the expert's command is. Driving follows the most likely candidate, so two
                          different expert answers (left / right round an obstacle) are not averaged into one
    --modes               two wheel heads, "target" (drive to the parking spot) and "evade" (get out of a mover's
                          way), and a learned switch (one logit: evade or not). Each head is trained only on rows of
                          its own mode (picked by the true label); the switch on the label by cross-entropy. Driving
                          uses the switch with hysteresis (eval_obstacles.LearnedBase). Not with --mixture
    --mode-label SOURCE   where the evade label comes from: "collision" (default; worked out from saved data, see
                          collision_labels) or "expert" (a per-row "expert_evade" saved by a teacher; error if absent)
    --validation FOLDER   dataset for the validation loss (default runs/obst-v2b)
    --balance             weighted loss: rows from hard layouts (scenario id contains "hard") and normal rows
                          count equally in total, however many rows each group has
    --out FOLDER          where to save model.pt
The saved weights are those of the epoch with the lowest validation loss (early stopping), not the last one.

Example:
    python train_base_torch.py --data runs/obst-v2b runs/obst-v3-extra-0 --no-velocity --out runs/my-model
"""
import argparse
import json
import re
from pathlib import Path
import numpy as np
import torch
from torch import nn
from models import base_inputs, BASE_INPUT_NAMES
from obstacle_env import ROBOT_RADIUS

ROOT = Path(__file__).resolve().parent
DATA = ROOT/"runs"/"obst-v2b"
OUT = ROOT/"runs"/"obst-base-torch-v1"
STRIDE = 10            # keep every 10th row: neighbouring 1 ms rows are near-copies
SCAN_GAP_S = 0.5       # seconds between stacked scans: over 1 s a 0.2 m/s cylinder moves 20 cm, enough to show
HIDDEN = 64
EPOCHS = 30
BATCH = 256
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INPUT_NAMES = BASE_INPUT_NAMES + [f"lidar_{10*k}deg_m" for k in range(36)]   # 36 rays, one scan
VELOCITY = ("v_forward_m_s", "v_left_m_s", "yaw_rate_rad_s")
WINDOW, FILTERS, SECTORS = 5, (8, 4), 12     # ray filter: 5 neighbouring rays, 8 then 4 detectors, 12 sectors of 30 deg
THREAT_S, THREAT_M = 3.0, 1.0                # threat = a mover comes within 1 m of the robot within the next 3 s
THREAT_NAMES = ["threat", "mover_forward_m", "mover_left_m", "mover_v_forward_m_s", "mover_v_left_m_s"]
THREAT_WEIGHT = 0.5                          # share of the threat loss next to the wheel loss
LOG_STD = (-4.0, 1.0)                        # mixture: limits on each candidate's log spread (scaled units)
# Modes: a row is "evade" when, both at constant velocity, the robot's centre and a mover's centre come closer than
# ROBOT_RADIUS + the mover's radius + MODE_MARGIN_M within MODE_HORIZON_S (checked every MODE_STEP_S)
MODE_HORIZON_S, MODE_MARGIN_M, MODE_STEP_S = 3.0, 0.3, 0.05
SWITCH_WEIGHT = 1.0                          # share of the switch's cross-entropy next to the wheel loss
# Menu output: the dynamic window teacher's options (obstacle_env.WindowExpert speeds x turns, 7 x 9 = 63) and "hold"
# (exactly still: the expert's command is exactly 0 when it waits or has parked). The model picks one and adds a small
# correction to it, measured in menu steps (so parking's off-menu commands still fit); hold has no correction
MENU_SPEEDS = (-0.1, 0.0, 0.05, 0.1, 0.15, 0.2, 0.25)          # m/s
MENU_TURNS = (-0.6, -0.45, -0.3, -0.15, 0.0, 0.15, 0.3, 0.45, 0.6)  # rad/s
MENU_STEP = (0.05, 0.15)                     # one menu step in speed and turn: the unit of the correction
MENU_FIX_WEIGHT = 1.0                        # share of the correction's squared error next to the choice's cross-entropy
WHEEL_RADIUS, TRACK = 0.11, 0.66             # skid steer, as obstacle_env (wheels: front right, front left, back left, back right)
HOLD_HIDDEN = 32                             # hold output (train_hold.py): neurons of its one hidden layer
PATIENCE = 8                                 # --patience: stop after this many epochs without a better validation loss
# Mirror flip (--mirror, 5 Nov): the same moment seen left-right mirrored. Leftward inputs change sign, each ray swaps
# with its mirror ray, the left and right wheels swap, and the threat's leftward position and velocity change sign.
MIRROR_NEGATE = ("left_m", "sin_heading_err", "v_left_m_s", "yaw_rate_rad_s")
MIRROR_WHEELS = [1, 0, 3, 2]                 # [FR, FL, BL, BR] -> [FL, FR, BR, BL]
MIRROR_THREAT = np.array([1, 1, -1, 1, -1], np.float32)    # sign per THREAT_NAMES entry


def mirror_map(names):
    """Left-right mirror of input rows with these column names: (column each one takes its value from, sign).
    The ray at angle a swaps with the ray at -a (same scan); the MIRROR_NEGATE inputs change sign."""
    index = {n: i for i, n in enumerate(names)}
    perm, sign = [], []
    for n in names:
        ray = re.fullmatch(r"(.*_)([0-9.]+)deg_m", n)
        perm.append(index[f"{ray[1]}{(360 - float(ray[2])) % 360:g}deg_m"] if ray else index[n])
        sign.append(-1.0 if n in MIRROR_NEGATE else 1.0)
    return np.array(perm), np.array(sign, np.float32)


def mirror_symmetric(mean, std, perm, sign):
    """Scaling statistics of the rows together with their mirror images. With these, mirroring a SCALED row is the
    same column swap and sign change as mirroring the raw row, so batches can be flipped after scaling."""
    second = std.astype(np.float64)**2 + mean.astype(np.float64)**2
    mean = (mean + sign*mean[perm])/2
    std = np.sqrt(np.maximum((second + second[perm])/2 - mean**2, 1e-12))
    return mean.astype(np.float32), std.astype(np.float32)


def filter_sizes(rays):
    """Ray filter window and pooling (in rays) for a lidar with this many rays per scan. They cover the same ANGLES as
    the 36-ray design whatever the ray count: the window spans about 40 deg (odd, so it stays centred on its ray:
    5 rays at 10 deg, 9 at 5 deg) and the pooling 30 deg (3 rays at 10 deg, 6 at 5 deg), so there are always SECTORS
    sectors and the trunk keeps its size."""
    if rays % SECTORS:
        raise ValueError(f"the ray filter needs a multiple of {SECTORS} rays per scan, got {rays}")
    return 2*round(2*rays/36) + 1, rays // SECTORS


def menu_options():
    """(64, 2) speed and turn of every menu option, speeds outer: option k < 63 = (MENU_SPEEDS[k // 9], MENU_TURNS[k % 9]);
    the last one (HOLD) is "exactly still"."""
    return np.array([(v, w) for v in MENU_SPEEDS for w in MENU_TURNS] + [(0.0, 0.0)], np.float32)


HOLD = len(MENU_SPEEDS)*len(MENU_TURNS)      # index of the hold option


def speed_turn(wheels):
    """Wheel speeds (rows, 4) in rad/s -> forward speed (m/s) and turn rate (rad/s), skid steer."""
    left, right = (wheels[:, 1] + wheels[:, 2])/2, (wheels[:, 0] + wheels[:, 3])/2
    return WHEEL_RADIUS*(left + right)/2, WHEEL_RADIUS*(right - left)/TRACK


def menu_labels(wheels):
    """Expert wheel speeds (rows, 4), rad/s -> nearest menu option (rows,) and the correction from it to the expert's
    command (rows, 2) in menu steps (0 for a command on the menu). An exactly-zero command is HOLD."""
    v, w = speed_turn(np.asarray(wheels, np.float64))
    options, step = menu_options()[:HOLD].astype(np.float64), np.array(MENU_STEP)
    gap = (np.stack([v, w], 1)[:, None, :] - options[None])/step         # (rows, 63, 2) in menu steps
    choice = (gap**2).sum(axis=2).argmin(axis=1)
    fix = gap[np.arange(len(v)), choice]
    still = (np.abs(v) < 1e-9) & (np.abs(w) < 1e-9)
    choice[still], fix[still] = HOLD, 0.0
    return choice, fix.astype(np.float32)


class RayNet(nn.Module):
    """Base network with a learned ray filter (encoder "conv"), a training-only threat output, a mixture output,
    two wheel heads with a switch (modes) or a menu output, and an optional hold output (train_hold.py: "the expert
    commands exactly 0 here", used only by LearnedBase). Input: base features, then `rays` rays per scan (current
    scan first), already scaled. forward() = 4 wheel speeds (scaled like the training targets)."""
    def __init__(self, n_base, scans, hidden, encoder="conv", threat=False, mixture=0, rays=36, modes=False, menu=False,
                 hold=False, dropout=0.0):
        super().__init__()
        if sum(map(bool, (mixture, modes, menu))) > 1:
            raise ValueError("--modes, --mixture and --menu cannot be combined")
        self.n_base, self.scans, self.encoder, self.rays_per_scan = n_base, scans, encoder, rays
        if encoder == "conv":
            window, pool = filter_sizes(rays)
            pad = window // 2
            self.rays = nn.Sequential(   # circular: the window wraps from the last ray round to the first
                nn.Conv1d(scans, FILTERS[0], window, padding=pad, padding_mode="circular"), nn.ReLU(),
                nn.Conv1d(FILTERS[0], FILTERS[1], window, padding=pad, padding_mode="circular"), nn.ReLU(),
                nn.MaxPool1d(pool), nn.Flatten())
            n_lidar = FILTERS[1]*SECTORS
        else:
            self.rays = nn.Identity()
            n_lidar = rays*scans
        # dropout 0: exactly the old layers (saved models load unchanged); otherwise dropout after both trunk layers
        layers = [nn.Linear(n_base + n_lidar, hidden), nn.ReLU()] + ([nn.Dropout(dropout)] if dropout else []) \
            + [nn.Linear(hidden, hidden), nn.ReLU()] + ([nn.Dropout(dropout)] if dropout else [])
        self.trunk = nn.Sequential(*layers)
        self.mixture, self.modes, self.menu = mixture, modes, menu
        # mixture: per candidate a score, 4 wheel speeds and 4 log spreads; modes: 4 target + 4 evade wheel speeds;
        # menu: per option a score and a (speed, turn) correction
        n_menu = HOLD + 1
        self.wheels = nn.Linear(hidden, mixture*9 if mixture else 8 if modes else 3*n_menu if menu else 4)
        if menu:    # forward() returns wheel speeds scaled like the training targets: train() fills these in
            self.register_buffer("options", torch.tensor(menu_options()))
            self.register_buffer("y_mean", torch.zeros(4))
            self.register_buffer("y_std", torch.ones(4))
        self.switch = nn.Linear(hidden, 1) if modes else None
        self.threat = nn.Linear(hidden, len(THREAT_NAMES)) if threat else None
        self.hold = nn.Sequential(nn.Linear(hidden, HOLD_HIDDEN), nn.ReLU(), nn.Linear(HOLD_HIDDEN, 1)) if hold else None

    def features(self, x):
        base, lidar = x[:, :self.n_base], x[:, self.n_base:]
        if self.encoder == "conv":
            lidar = lidar.reshape(len(x), self.scans, self.rays_per_scan)    # scans become the filter's channels
        return self.trunk(torch.cat([base, self.rays(lidar)], dim=1))

    def candidates(self, h):
        """Mixture output from features h -> log probabilities (rows, K), wheel speeds and log spreads (rows, K, 4)."""
        out = self.wheels(h)
        k = self.mixture
        log_p = torch.log_softmax(out[:, :k], dim=1)
        mean = out[:, k:5*k].reshape(-1, k, 4)
        log_std = out[:, 5*k:].reshape(-1, k, 4).clamp(*LOG_STD)
        return log_p, mean, log_std

    def choices(self, h):
        """Menu output from features h -> option scores (rows, 64) and corrections (rows, 64, 2) in menu steps."""
        out = self.wheels(h)
        n = len(self.options)
        return out[:, :n], out[:, n:].reshape(-1, n, 2)

    def menu_wheels(self, h):
        """The highest-scoring option plus its correction (none for hold), as scaled wheel speeds (rows, 4)."""
        score, fix = self.choices(h)
        k = score.argmax(dim=1)
        rows = torch.arange(len(k), device=k.device)
        fix = fix[rows, k]*torch.tensor(MENU_STEP, device=k.device)*(k != HOLD)[:, None]
        v, w = (self.options[k] + fix).unbind(dim=1)
        left, right = (v - w*TRACK/2)/WHEEL_RADIUS, (v + w*TRACK/2)/WHEEL_RADIUS
        return (torch.stack([right, left, left, right], dim=1) - self.y_mean)/self.y_std

    def heads(self, h):
        """Modes output from features h -> switch logit (rows,) (evade when above 0) and both heads' wheel speeds
        (rows, 2, 4): [:, 0] target head, [:, 1] evade head."""
        return self.switch(h)[:, 0], self.wheels(h).reshape(-1, 2, 4)

    def forward(self, x):
        h = self.features(x)
        if self.modes:      # the head the switch prefers at this moment (LearnedBase adds hysteresis while driving)
            logit, both = self.heads(h)
            return both[torch.arange(len(x)), (logit > 0).long()]
        if self.menu:       # follow one option of the menu, never a blend of them
            return self.menu_wheels(h)
        if not self.mixture:
            return self.wheels(h)
        log_p, mean, _ = self.candidates(h)      # follow the most likely candidate, never a blend of them
        return mean[torch.arange(len(x)), log_p.argmax(dim=1)]


def mixture_nll(log_p, mean, log_std, y):
    """Negative log-likelihood of the expert's wheel speeds y (rows, 4) under the mixture, per row: low when ONE
    candidate is close to y, whatever the others say, so different expert answers can each keep a candidate."""
    z = (y[:, None, :] - mean)/log_std.exp()
    log_like = (-0.5*z**2 - log_std - 0.5*np.log(2*np.pi)).sum(dim=2)    # (rows, K), diagonal Gaussian each
    return -torch.logsumexp(log_p + log_like, dim=1)


def menu_loss(score, fix, choice, gap):
    """Per row: cross-entropy of the expert's nearest option + MENU_FIX_WEIGHT x squared error of that option's
    correction (menu steps)."""
    rows = torch.arange(len(choice), device=choice.device)
    return (nn.functional.cross_entropy(score, choice, reduction="none")
            + MENU_FIX_WEIGHT*((fix[rows, choice] - gap)**2).sum(dim=1))


def build_net(n_in, hidden, scans=1, encoder="mlp", threat=False, mixture=0, rays=36, modes=False, menu=False,
              hold=False, dropout=0.0):
    """The plain network exactly as before (old model.pt files load unchanged), or a RayNet."""
    if encoder == "mlp" and not threat and not mixture and not modes and not menu and not hold:
        return nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 4))
    return RayNet(n_in - rays*scans, scans, hidden, encoder, threat, mixture, rays, modes, menu, hold, dropout)


def threat_labels(z, moving, stride):
    """(rows, 5) per saved row: [1 if a mover comes within THREAT_M of the robot's actual path in the next THREAT_S,
    that mover's position and velocity in the robot's directions now]. Zeros when there is no threat.
    z: one trajectory.npz; moving: which obstacles move (summary.json)."""
    obs = z["observations"][::stride]
    out = np.zeros((len(obs), len(THREAT_NAMES)), np.float32)
    moving = np.asarray(moving, bool)
    if "obstacle_xy" not in z.files or not moving.any():
        return out
    dt = STRIDE/1000                                   # loaded rows are STRIDE ms apart in every dataset
    xy = z["obstacle_xy"][::stride][:, moving]         # (rows, movers, 2)
    rel = xy - z["base_m"][::stride][:, None, :2]
    dist = np.linalg.norm(rel, axis=2)
    ahead = int(round(THREAT_S/dt))
    padded = np.concatenate([dist, np.repeat(dist[-1:], ahead, axis=0)])    # after the end: stays where it was
    soonest = np.lib.stride_tricks.sliding_window_view(padded, ahead + 1, axis=0).min(axis=2)   # (rows, movers)
    which = soonest.argmin(axis=1)
    rows = np.arange(len(obs))
    threat = soonest[rows, which] < THREAT_M
    speed = np.gradient(xy, dt, axis=0) if len(xy) > 1 else np.zeros_like(xy)
    yaw = np.arctan2(obs[:, 3], obs[:, 4])
    c, s = np.cos(yaw), np.sin(yaw)
    p, v = rel[rows, which], speed[rows, which]
    frame = np.stack([c*p[:, 0] + s*p[:, 1], c*p[:, 1] - s*p[:, 0], c*v[:, 0] + s*v[:, 1], c*v[:, 1] - s*v[:, 0]], axis=1)
    out[threat, 0] = 1.0
    out[threat, 1:] = frame[threat]
    return out


def collision_labels(z, moving, radii, stride):
    """(rows,) bool per saved row: True = "evade". Both kept at constant velocity, the robot's centre and some moving
    obstacle's centre come closer than ROBOT_RADIUS + its radius + MODE_MARGIN_M within the next MODE_HORIZON_S.
    Present information only, never the robot's or a mover's actual future path: the robot keeps its current velocity
    (observation columns 22-23, world frame) turning at its current yaw rate (column 27); each mover keeps the
    velocity of its last step (obstacle_xy, backward difference; the first row borrows the second's).
    z: one trajectory.npz; moving, radii: per obstacle (summary.json, the scenario in episode.json)."""
    obs = z["observations"][::stride]
    moving = np.asarray(moving, bool)
    if "obstacle_xy" not in z.files or not moving.any():
        return np.zeros(len(obs), bool)
    t = z["time_s"][::stride] if "time_s" in z.files else np.arange(len(obs))*STRIDE/1000
    xy = z["obstacle_xy"][::stride][:, moving]                     # (rows, movers, 2)
    v_mover = np.zeros_like(xy)
    if len(xy) > 1:
        v_mover[1:] = np.diff(xy, axis=0) / np.diff(t)[:, None, None]
        v_mover[0] = v_mover[1]
    tau = np.arange(0.0, MODE_HORIZON_S + 1e-9, MODE_STEP_S)       # (steps,) seconds ahead
    w = obs[:, 27]
    still_turning = np.abs(w) < 1e-6                               # straight line: the formulas below divide by w
    w_safe = np.where(still_turning, 1.0, w)[:, None]
    wt = w[:, None]*tau
    along = np.where(still_turning[:, None], tau, np.sin(wt)/w_safe)                    # integral of cos(w s)
    across = np.where(still_turning[:, None], 0.5*w[:, None]*tau**2, (1 - np.cos(wt))/w_safe)   # of sin(w s)
    vx, vy = obs[:, 22:23], obs[:, 23:24]                          # velocity turns at the yaw rate
    robot = obs[:, None, 0:2] + np.stack([along*vx - across*vy, across*vx + along*vy], axis=2)   # (rows, steps, 2)
    movers = xy[:, None] + tau[None, :, None, None]*v_mover[:, None]                            # (rows, steps, movers, 2)
    closest = np.linalg.norm(robot[:, :, None] - movers, axis=3).min(axis=1)                   # (rows, movers)
    return np.any(closest < ROBOT_RADIUS + np.asarray(radii, float)[moving] + MODE_MARGIN_M, axis=1)


def mode_labels(z, source, folder, ep, stride):
    """Evade label per saved row from `source`: "collision" (collision_labels) or "expert" (the teacher's own
    per-row "expert_evade" in trajectory.npz; an error if this run has none)."""
    if source == "expert":
        if "expert_evade" not in z.files:
            raise KeyError(f"{folder/ep['path']}: no expert_evade in trajectory.npz (use --mode-label collision)")
        return z["expert_evade"][::stride].astype(bool)
    episode = json.loads((folder/ep["path"]/"episode.json").read_text())
    radii = [o["r"] for o in episode["scenario"]["obstacles"]]
    moving = ep.get("obstacle_is_moving")
    if moving is None:
        moving = json.loads((folder/ep["path"]/"summary.json").read_text()).get("obstacle_is_moving", [])
    return collision_labels(z, moving, radii, stride)


def lidar_names(rays=36, prefix="lidar_"):
    """Names of one scan's rays, on the 36-ray pattern: lidar_0deg_m, lidar_10deg_m, ... (72 rays: every 5 deg)."""
    return [f"{prefix}{360*j/rays:g}deg_m" for j in range(rays)]


TARGET_RAY_NAMES = ["target_ray_m", "target_ray_gap_m"]


def target_ray(base, lidar):
    """The "target blocked?" inputs (5 Nov) from the base features (rows, 9: forward_m, left_m, ...) and the CURRENT scan
    (rows, rays): how far the lidar sees in the target's direction (interpolated between the two nearest rays) and that
    minus the target's horizontal distance (negative: something stands between the robot and the target). (rows, 2)"""
    base, lidar = np.atleast_2d(base), np.atleast_2d(lidar)
    rays = lidar.shape[1]
    bearing = np.mod(np.arctan2(base[:, 1], base[:, 0]), 2*np.pi)
    pos = bearing/(2*np.pi/rays)
    i0 = np.floor(pos).astype(int) % rays
    frac = pos - np.floor(pos)
    rows = np.arange(len(lidar))
    seen = (1 - frac)*lidar[rows, i0] + frac*lidar[rows, (i0 + 1) % rays]
    return np.stack([seen, seen - np.hypot(base[:, 0], base[:, 1])], axis=1).astype(np.float32)


def input_names(scans=1, rays=36, target_ray=False):
    """Base features, [the target-ray inputs,] the current scan's rays, then the rays of each earlier scan.
    INPUT_NAMES for 1 scan of 36."""
    return (BASE_INPUT_NAMES + (TARGET_RAY_NAMES if target_ray else []) + lidar_names(rays)
            + [n for k in range(1, scans) for n in lidar_names(rays, f"lidar_t-{k*SCAN_GAP_S}s_")])


def dataset_range(folder):
    """Lidar range (m) of a dataset folder: manifest "lidar_range" (4 m for folders saved before it existed, 5 Nov)."""
    return float(json.loads((Path(folder)/"manifest.json").read_text()).get("lidar_range", 4.0))


def dataset_rays(folder):
    """Lidar rays per scan in a dataset folder: manifest "lidar_rays" (36 for folders saved before it existed)."""
    return int(json.loads((Path(folder)/"manifest.json").read_text()).get("lidar_rays", 36))


def stack_scans(lidar, scans, gap_rows):
    """lidar (rows, rays) -> (rows, rays*scans): the current scan, then each earlier one. Before the episode's start
    there is no scan yet, so the first one repeats, as LearnedBase does while driving."""
    rows = np.arange(len(lidar))
    return np.hstack([lidar[np.maximum(rows - k*gap_rows, 0)] for k in range(scans)])


def set_stride(ms):
    """Rows used for training are `ms` apart (default STRIDE = 10). Data is saved every 1 or 10 ms, so ms must be a
    multiple of 10; the scan gap and the threat labels' time step follow it."""
    global STRIDE
    if ms % 10 or round(SCAN_GAP_S*1000) % ms:
        raise ValueError(f"stride_ms must be a multiple of 10 that divides the {SCAN_GAP_S*1000:g} ms between scans "
                         "(else training scans would sit at a different time gap than when driving)")
    STRIDE = int(ms)


def group_of(scenario_id):
    """Room type of a scenario for balance_groups: its id up to the first "_" (e.g. "ovc3_0012" -> "ovc3")."""
    return scenario_id.split("_")[0]


ROW_CACHE = True        # load(): read each episode from its rows_<STRIDE>ms.npz cache (built once, see strided_rows)
CACHE_WORKERS = 12      # processes that build missing caches
CACHE_KEYS = ("observations", "lidar", "lidar_seen", "actions", "expert_wheels", "obstacle_xy", "base_m")


def cache_path(folder, ep):
    return Path(folder)/ep["path"]/f"rows_{STRIDE}ms.npz"


def strided_rows(job):
    """Write one episode's arrays at the training stride, exactly as load() slices them, to an uncompressed
    rows_<STRIDE>ms.npz beside trajectory.npz (6 Nov: reading the full-rate compressed files took ~0.6 s per run,
    and DAgger reads every teacher run again each round). job = (episode folder, stride, cache file)."""
    path, stride, out = job
    z = np.load(path/"trajectory.npz")
    part = out.with_name(out.stem + ".part.npz")
    np.savez(part, **{k: z[k][::stride] for k in CACHE_KEYS if k in z.files})
    part.replace(out)       # whole files only: an interrupted build leaves no half cache behind


def build_caches(folder, episodes, stride):
    """Missing row caches of these episodes, built in parallel."""
    jobs = [(Path(folder)/ep["path"], stride, cache_path(folder, ep)) for ep in episodes if not cache_path(folder, ep).exists()]
    if len(jobs) > 1 and CACHE_WORKERS > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(min(CACHE_WORKERS, len(jobs))) as pool:
            list(pool.map(strided_rows, jobs, chunksize=4))
    else:
        for job in jobs:
            strided_rows(job)


def load(split, folder=DATA, label="actions", scans=1, threat=False, mode_label=None, groups=None, target_ray_=False):
    """All usable episodes of one split in one dataset folder -> inputs (rows, 9 + rays*scans), wheel targets (rows, 4)
    and a hard-layout flag per row (rows,). threat=True: also the threat labels (rows, 5) (threat_labels).
    mode_label "collision" / "expert": also (last) the evade label per row (rows,) bool (mode_labels).
    rays = the folder's manifest "lidar_rays" (dataset_rays); every episode's lidar must have that width.

    label "actions": an expert dataset; only successful episodes, target = the expert's own wheel commands.
    label "expert_wheels": a DAgger dataset; every episode (the learned policy may have failed, that is the
    point), target = what the expert would have done in each state the policy visited.
    groups: a list; if given, the room type of every row (group_of) is appended to it, one array per episode.
    """
    folder = Path(folder)
    manifest = json.loads((folder/"manifest.json").read_text())
    stride = max(1, STRIDE // manifest.get("saved_every", 1))   # DAgger runs are already saved every 10th step
    rays = dataset_rays(folder)
    xs, ys, hard, threats, modes = [], [], [], [], []
    episodes = [ep for ep in manifest["episodes"]
                if ep["split"] == split and (label != "actions" or ep["success"])]
    cached = ROW_CACHE and not mode_label       # mode labels read other arrays: those load the full file
    if cached:
        build_caches(folder, episodes, stride)
    for ep in episodes:
        z = np.load(cache_path(folder, ep)) if cached else np.load(folder/ep["path"]/"trajectory.npz")
        step = 1 if cached else stride          # cached rows are already STRIDE ms apart
        obs = z["observations"][::step]
        # DAgger runs with noisy sensors: train on what the policy SAW, labelled by what the expert would do
        lidar = z["lidar_seen" if label == "expert_wheels" and "lidar_seen" in z else "lidar"][::step]
        wheels = z["actions"][::step, :4] if label == "actions" else z["expert_wheels"][::step]
        if lidar.shape[1] != rays:
            raise ValueError(f"{folder/ep['path']}: lidar has {lidar.shape[1]} rays, the manifest says {rays}")
        base = np.array([base_inputs(row) for row in obs])      # 9 base features for every row
        if target_ray_:
            base = np.hstack([base, target_ray(base, lidar)])   # + "target blocked?" from the current scan
        lidar = stack_scans(lidar, scans, int(round(SCAN_GAP_S*1000/STRIDE)))   # rows are STRIDE ms apart
        xs.append(np.hstack([base, lidar]).astype(np.float32))   # + rays per scan -> (rows, 45) for one scan of 36
        ys.append(wheels.astype(np.float32))                     # float32 per episode: no float64 copy of everything
        hard.append(np.full(len(wheels), "hard" in ep.get("scenario_id", ep["path"])))
        if groups is not None:
            groups.append(np.full(len(wheels), group_of(ep.get("scenario_id", ep["path"]))))
        if threat:
            moving = ep.get("obstacle_is_moving")
            if moving is None:
                moving = json.loads((folder/ep["path"]/"summary.json").read_text()).get("obstacle_is_moving", [])
            threats.append(threat_labels(z, moving, step))
        if mode_label:
            modes.append(mode_labels(z, mode_label, folder, ep, stride))
    if not xs:
        out = np.zeros((0, len(input_names(scans, rays, target_ray_))), np.float32), np.zeros((0, 4), np.float32), np.zeros(0, bool)
        threats, modes = [np.zeros((0, len(THREAT_NAMES)), np.float32)], [np.zeros(0, bool)]
    else:
        out = np.concatenate(xs), np.concatenate(ys), np.concatenate(hard)
    return out + ((np.concatenate(threats),) if threat else ()) + ((np.concatenate(modes),) if mode_label else ())


def noisy_copy(x, rng, noise_max, dropout_max, rays=36):
    """Copy of input rows with lidar noise and dropouts (each row gets its own random strength up to the max).
    The labels stay the expert's clean-sensor actions: "this is what you might see; this is still the right move".
    Current scan only (rays per scan)."""
    x = x.copy()
    lid = slice(len(BASE_INPUT_NAMES), len(BASE_INPUT_NAMES) + rays)
    n = len(x)
    sigma = rng.uniform(0, noise_max, (n, 1))
    drop = rng.random((n, rays)) < rng.uniform(0, dropout_max, (n, 1))
    noisy = x[:, lid] + sigma*rng.standard_normal((n, rays))
    noisy[drop] = 4.0
    x[:, lid] = np.clip(noisy, 0.5, 4.0)
    return x


def train(data=(DATA,), dagger=(), out=OUT, no_velocity=False, epochs=EPOCHS, seed=0, validation=DATA,
          lidar_noise=0.0, lidar_dropout=0.0, balance=False, scans=1, hidden=HIDDEN, encoder="mlp", threat=False,
          mixture=0, modes=False, mode_label="collision", menu=False, evade_weight=1.0, mirror=False, patience=None,
          balance_groups=False, target_ray=False, dropout=0.0, weight_decay=0.0):
    """Train one model and save it to out/model.pt. Returns the validation wheel RMSE (rad/s).
    lidar_noise / lidar_dropout > 0: add a noisy copy of the EXPERT rows (noise-robust training).
    balance: weight the loss so hard-layout rows and normal rows count equally in total.
    encoder "conv": learned ray filter (RayNet); threat: add the training-only threat output.
    mixture K > 0: K candidate wheel commands trained on likelihood (validation loss is then the likelihood too).
    modes: target and evade wheel heads + a switch, labels from mode_label (validation loss = wheel loss with the head
    picked by the true label + the switch's cross-entropy). The lidar ray count comes from the datasets.
    menu: pick one of the window teacher's 63 options + a correction (cross-entropy + squared error; validation loss
    the same). evade_weight W != 1: rows whose mode_label is "evade" count W times (the mean weight stays 1).
    mirror: every batch shows a random half of its rows left-right mirrored (mirror_map; validation rows unchanged).
    patience N: stop after N epochs without a better validation loss (the best epoch is kept, as always).
    target_ray: two extra inputs, the lidar range towards the target and that minus the target distance (target_ray()).
    dropout P / weight_decay W (Julia's fix for overfitting in the first epoch): dropout after both trunk layers, AdamW.
    balance_groups: every room type (group_of the scenario id) gets the same total loss weight, whatever its row count
    (5 Nov: long floor-plan runs must not drown short open-room runs). Not combined with balance."""
    out = Path(out)
    if out.exists():
        raise FileExistsError(f"Choose a new output folder: {out}")
    if sum(map(bool, (mixture, modes, menu))) > 1:
        raise ValueError("--modes, --mixture and --menu cannot be combined")
    if mirror and menu:
        raise ValueError("--mirror does not mirror the menu labels")
    folder_rays = {str(f): dataset_rays(f) for f in [*data, *dagger, validation]}
    if len(set(folder_rays.values())) > 1:
        raise ValueError(f"datasets with different lidar ray counts cannot be mixed: {folder_rays}")
    rays = next(iter(folder_rays.values()))
    folder_range = {str(f): dataset_range(f) for f in [*data, *dagger, validation]}
    if len(set(folder_range.values())) > 1:
        raise ValueError(f"datasets with different lidar ranges cannot be mixed: {folder_range}")
    lidar_range = next(iter(folder_range.values()))
    labels = mode_label if modes or evade_weight != 1 else None    # evade label per row: needed for modes and weighting
    if scans > 1 and (lidar_noise > 0 or lidar_dropout > 0):
        raise ValueError("noisy_copy only knows the current scan")
    if (lidar_noise > 0 or lidar_dropout > 0) and threat:
        raise ValueError("noisy copies have no threat labels yet")
    if balance and balance_groups:
        raise ValueError("balance and balance_groups cannot be combined")
    torch.manual_seed(seed)
    groups = [] if balance_groups else None
    parts = ([load("train", f, scans=scans, threat=threat, mode_label=labels, groups=groups, target_ray_=target_ray) for f in data]
             + [load("train", f, "expert_wheels", scans, threat, labels, groups=groups, target_ray_=target_ray) for f in dagger])
    x, y = np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])
    hard = np.concatenate([p[2] for p in parts])
    t = np.concatenate([p[3] for p in parts]) if threat else None
    m = np.concatenate([p[-1] for p in parts]) if labels else None
    n_expert = sum(len(p[0]) for p in parts[:len(data)])
    del parts               # memory: the joined arrays are all that is needed from here
    if lidar_noise > 0 or lidar_dropout > 0:
        x = np.concatenate([x, noisy_copy(x[:n_expert], np.random.default_rng(seed), lidar_noise, lidar_dropout, rays)])
        y = np.concatenate([y, y[:n_expert]])
        hard = np.concatenate([hard, hard[:n_expert]])
        if m is not None:
            m = np.concatenate([m, m[:n_expert]])
    # Weighted loss: each group gets half the total weight; the mean weight stays 1
    weights = np.ones(len(x), np.float32)
    if balance and 0 < hard.sum() < len(hard):
        weights[hard] = len(x) / (2*hard.sum())
        weights[~hard] = len(x) / (2*(~hard).sum())
    group_rows = {}
    if balance_groups:
        g = np.concatenate(groups)
        if len(g) != len(x):
            raise ValueError("balance_groups does not work with noisy copies")
        names_g, counts = np.unique(g, return_counts=True)
        group_rows = dict(zip(names_g.tolist(), counts.tolist()))
        for name, count in group_rows.items():
            weights[g == name] = len(x) / (len(group_rows)*count)
        del g, groups
    if evade_weight != 1:
        weights[m] *= evade_weight
        weights *= len(weights)/weights.sum()
    weighted = balance or evade_weight != 1 or balance_groups
    val = load("validation", validation, scans=scans, mode_label=labels, target_ray_=target_ray)
    xv, yv = val[:2]
    names = input_names(scans, rays, target_ray)
    columns = [i for i, n in enumerate(names) if not (no_velocity and n in VELOCITY)]
    x, xv = x[:, columns], xv[:, columns]
    if mirror:
        perm, sign = mirror_map([names[i] for i in columns])
    print(f"train rows {len(x)} ({len(data)} expert + {len(dagger)} DAgger folders, {int(hard.sum())} from hard "
          f"layouts), validation rows {len(xv)}, inputs {x.shape[1]} ({rays} rays per scan), device {DEVICE}")
    if modes:
        print(f"evade rows ({mode_label}): train {100*m.mean():.1f}%, validation {100*val[-1].mean():.1f}%")

    # Scaling (what StandardScaler did for you in sklearn): learn it on TRAIN only
    x_mean, x_std = x.mean(axis=0), x.std(axis=0) + 1e-6
    y_mean, y_std = y.mean(axis=0), y.std(axis=0) + 1e-6
    if encoder == "conv":   # one shared scale for every ray, so the filter sees the same numbers at every angle
        lid = slice(x.shape[1] - rays*scans, x.shape[1])
        x_mean[lid], x_std[lid] = x[:, lid].mean(), x[:, lid].std() + 1e-6
    if mirror:              # statistics of the data plus its mirror image, so the flip is exact after scaling
        x_mean, x_std = mirror_symmetric(x_mean, x_std, perm, sign)
        y_mean, y_std = mirror_symmetric(y_mean, y_std, np.array(MIRROR_WHEELS), np.ones(4, np.float32))
    if threat:              # position and velocity scaled on the rows that have a threat; the flag stays 0/1
        flagged = t[:, 0] > 0.5
        t_mean, t_std = np.zeros(len(THREAT_NAMES), np.float32), np.ones(len(THREAT_NAMES), np.float32)
        if flagged.any():
            t_mean[1:], t_std[1:] = t[flagged, 1:].mean(axis=0), t[flagged, 1:].std(axis=0) + 1e-6
        if mirror:
            t_mean, t_std = mirror_symmetric(t_mean, t_std, np.arange(len(THREAT_NAMES)), MIRROR_THREAT)
        T = torch.tensor((t - t_mean)/t_std, device=DEVICE)
        print(f"threat rows {int(flagged.sum())} of {len(t)} ({100*flagged.mean():.1f}%)")

    def to_tensor(a):
        return torch.as_tensor(a, device=DEVICE)     # on the CPU it shares the array instead of copying it

    def scaled(a, mean, std):
        out = a - mean      # one new array, then scaled in place (memory: the training rows are several GB)
        out /= std
        return out
    x -= x_mean             # scaled in place: the input rows are the biggest array (several GB)
    x /= x_std
    X, Y, W = to_tensor(x), to_tensor(scaled(y, y_mean, y_std)), to_tensor(weights)
    XV, YV = to_tensor((xv - x_mean)/x_std), to_tensor((yv - y_mean)/y_std)
    if modes:
        M, MV = to_tensor(m.astype(np.float32)), to_tensor(val[-1].astype(np.float32))
    if menu:    # labels from the raw (unscaled) wheel speeds
        choice, gap = menu_labels(y)
        C, G = torch.as_tensor(choice, device=DEVICE), to_tensor(gap)
        choice, gap = menu_labels(yv)
        CV, GV = torch.as_tensor(choice, device=DEVICE), to_tensor(gap)
        print(f"menu: {100*float((np.abs(gap).sum(axis=1) < 1e-3).mean()):.1f}% of validation rows exactly on the menu")

    model = build_net(x.shape[1], hidden, scans, encoder, threat, mixture, rays, modes, menu, dropout=dropout).to(DEVICE)
    if menu:
        model.y_mean.copy_(torch.as_tensor(y_mean))
        model.y_std.copy_(torch.as_tensor(y_std))
    optimiser = (torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=weight_decay) if weight_decay
                 else torch.optim.Adam(model.parameters(), lr=LR))
    loss_fn = nn.MSELoss()
    switch_loss = nn.functional.binary_cross_entropy_with_logits

    def own_head(both, mode):
        """Modes: each row's wheel speeds from the head of its TRUE mode (teacher forcing)."""
        return both[torch.arange(len(both), device=both.device), mode.long()]

    def validation_loss():
        if mixture:
            return mixture_nll(*model.candidates(model.features(XV)), YV).mean().item()
        if menu:
            return menu_loss(*model.choices(model.features(XV)), CV, GV).mean().item()
        if modes:
            logit, both = model.heads(model.features(XV))
            return (loss_fn(own_head(both, MV), YV) + SWITCH_WEIGHT*switch_loss(logit, MV)).item()
        return loss_fn(model(XV), YV).item()

    if mirror:
        PERM, SIGN = torch.as_tensor(perm, device=DEVICE), torch.as_tensor(sign, device=DEVICE)
        TSIGN = torch.as_tensor(MIRROR_THREAT, device=DEVICE)
    best_loss, best_epoch, best_state = float("inf"), 0, None
    epochs_run = epochs
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(X), device=DEVICE)        # shuffle every epoch
        total = 0.0
        for start in range(0, len(X), BATCH):
            idx = order[start:start+BATCH]
            xb, yb = X[idx], Y[idx]
            tb = T[idx] if threat else None
            if mirror:    # a random half of the batch, seen left-right mirrored
                flip = (torch.rand(len(idx), device=DEVICE) < 0.5)[:, None]
                xb = torch.where(flip, xb[:, PERM]*SIGN, xb)
                yb = torch.where(flip, yb[:, MIRROR_WHEELS], yb)
                if threat:
                    tb = torch.where(flip, tb*TSIGN, tb)
            if threat or mixture or modes or menu:
                hidden_out = model.features(xb)
                guess = model.threat(hidden_out) if threat else None
            if mixture:   # per row: how unlikely the expert's command is under the candidates
                per_row = mixture_nll(*model.candidates(hidden_out), yb)
                loss = (per_row * W[idx]).mean() if weighted else per_row.mean()
            elif menu:    # per row: which option, and how far the expert's command is from it
                per_row = menu_loss(*model.choices(hidden_out), C[idx], G[idx])
                loss = (per_row * W[idx]).mean() if weighted else per_row.mean()
            else:
                if modes:     # each head only learns from rows of its own mode
                    logit, both = model.heads(hidden_out)
                    prediction = own_head(both, M[idx])
                else:
                    prediction = model.wheels(hidden_out) if threat else model(xb)
                if weighted:  # weighted mean squared error: each row's error counts by its weight
                    loss = (((prediction - yb)**2).mean(dim=1) * W[idx]).mean()
                else:
                    loss = loss_fn(prediction, yb)
                if modes:     # the switch: evade or not, cross-entropy on the label
                    loss = loss + SWITCH_WEIGHT*switch_loss(logit, M[idx])
            if threat:    # yes/no: cross-entropy; where and how fast: only on rows where there is a threat
                answer = tb
                aux = nn.functional.binary_cross_entropy_with_logits(guess[:, 0], answer[:, 0])
                flagged = answer[:, 0] > 0.5
                if flagged.any():
                    aux = aux + ((guess[flagged, 1:] - answer[flagged, 1:])**2).mean()
                loss = loss + THREAT_WEIGHT*aux
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            total += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            val_loss = validation_loss()
        if val_loss < best_loss:     # early stopping: remember the best epoch's weights
            best_loss, best_epoch = val_loss, epoch + 1
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        print(f"epoch {epoch+1:2d}  train loss {total/len(X):.4f}  validation loss {val_loss:.4f}")
        if patience and epoch + 1 - best_epoch >= patience:
            epochs_run = epoch + 1
            print(f"early stop: no better validation loss for {patience} epochs")
            break
    model.load_state_dict(best_state)

    # Validation RMSE in real units (rad/s)
    with torch.no_grad():
        pred = model(XV).cpu().numpy() * y_std + y_mean
    rmse = float(np.sqrt(np.mean((pred - yv)**2)))
    print(f"validation wheel RMSE: {rmse:.3f} rad/s (best epoch {best_epoch}), "
          f"parameters {sum(p.numel() for p in model.parameters())}")

    out.mkdir(parents=True)
    torch.save({"state_dict": model.state_dict(), "hidden": hidden, "scans": scans, "encoder": encoder, "threat": threat,
                "mixture": mixture, "rays": rays, "lidar_range": lidar_range, "target_ray": target_ray, "dropout": dropout, "modes": modes, "mode_label": labels, "menu": menu, "input_names": [names[i] for i in columns], "columns": columns,
                "x_mean": x_mean, "x_std": x_std, "y_mean": y_mean, "y_std": y_std, "mirror": mirror}, out/"model.pt")
    (out/"training.json").write_text(json.dumps({"validation_wheel_rmse_rad_s": rmse, "best_epoch": best_epoch,
        "train_rows": len(x), "epochs": epochs, "stride": STRIDE, "no_velocity": no_velocity, "seed": seed,
        "lidar_noise": lidar_noise, "lidar_dropout": lidar_dropout, "balance": balance, "scans": scans, "hidden": hidden,
        "encoder": encoder, "threat": threat, "mixture": mixture, "rays": rays, "lidar_range": lidar_range, "modes": modes, "mode_label": labels,
        "menu": menu, "evade_weight": evade_weight, "mirror": mirror, "patience": patience, "epochs_run": epochs_run,
        "balance_groups": balance_groups, "group_rows": group_rows, "stride_ms": STRIDE, "target_ray": target_ray,
        "dropout": dropout, "weight_decay": weight_decay,
        **({"evade_share_train": float(m.mean()), "evade_share_validation": float(val[-1].mean())} if modes else {}),
        "parameters": sum(p.numel() for p in model.parameters()),
        "hard_rows": int(hard.sum()), "data": [str(f) for f in data], "dagger": [str(f) for f in dagger], "validation": str(validation)},
        indent=1))
    print(f"saved {out/'model.pt'}")
    return rmse


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", nargs="+", default=[str(DATA)])
    parser.add_argument("--dagger", nargs="*", default=[])
    parser.add_argument("--no-velocity", action="store_true")
    parser.add_argument("--scans", type=int, default=1, help="lidar scans per input, SCAN_GAP_S apart")
    parser.add_argument("--hidden", type=int, default=HIDDEN, help="neurons per hidden layer")
    parser.add_argument("--encoder", choices=["mlp", "conv"], default="mlp", help="conv: learned ray filter")
    parser.add_argument("--threat", action="store_true", help="training-only threat output")
    parser.add_argument("--mixture", type=int, default=0, metavar="K", help="K candidate wheel commands (0: one)")
    parser.add_argument("--modes", action="store_true", help="target and evade wheel heads with a learned switch")
    parser.add_argument("--mode-label", choices=["collision", "expert"], default="collision",
                        help="evade label: worked out from saved data, or the teacher's saved expert_evade")
    parser.add_argument("--menu", action="store_true", help="pick one of the window teacher's 63 options or hold + a correction")
    parser.add_argument("--evade-weight", type=float, default=1.0, help="loss weight of rows labelled evade (--mode-label)")
    parser.add_argument("--validation", default=str(DATA), help="dataset for the validation loss")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lidar-noise", type=float, default=0.0, help="max noise (m) for a noisy copy of expert rows")
    parser.add_argument("--lidar-dropout", type=float, default=0.0, help="max dropout chance for that copy")
    parser.add_argument("--balance", action="store_true", help="hard-layout and normal rows count equally")
    parser.add_argument("--mirror", action="store_true", help="show a random half of every batch left-right mirrored")
    parser.add_argument("--patience", type=int, help=f"stop after this many epochs without a better validation loss (e.g. {PATIENCE})")
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--balance-groups", action="store_true", help="every room type (scenario id prefix) weighs the same")
    parser.add_argument("--target-ray", action="store_true", help="add the 'target blocked?' inputs (target_ray)")
    parser.add_argument("--dropout", type=float, default=0.0, help="dropout after both trunk layers (conv encoder)")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="AdamW weight decay (0: Adam as before)")
    parser.add_argument("--stride-ms", type=int, default=STRIDE, help="ms between training rows (10 = every saved 10 ms row)")
    args = parser.parse_args()
    set_stride(args.stride_ms)
    train(args.data, args.dagger, args.out, args.no_velocity, args.epochs, args.seed, args.validation,
          lidar_noise=args.lidar_noise, lidar_dropout=args.lidar_dropout, balance=args.balance,
          scans=args.scans, hidden=args.hidden, encoder=args.encoder, threat=args.threat, mixture=args.mixture,
          modes=args.modes, mode_label=args.mode_label, menu=args.menu, evade_weight=args.evade_weight,
          mirror=args.mirror, patience=args.patience, balance_groups=args.balance_groups, target_ray=args.target_ray,
          dropout=args.dropout, weight_decay=args.weight_decay)


if __name__ == "__main__":
    main()
