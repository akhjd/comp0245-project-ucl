# Demo layouts

Every pick below is the fully learned robot (learned base drives and parks, learned arm reaches) on scenarios no
model was trained or chosen on. Outcomes were checked on 30 Oct by replaying each one (the simulation is
deterministic, so the same command gives the same run). Run from the project folder; each opens the PyBullet
viewer at real-time speed. Add `--speed 0.5` to slow down, or swap `full` for `expert` to show the course expert
on the same layout.

```bash
P=/home/akhjd/ros2_ws/RoboEnv/.pixi/envs/default/bin/python
```

## 1. Normal layouts
| What it shows | Command | Outcome |
|---|---|---|
| Straight ahead, one obstacle (warm-up) | `$P -m comp0245.apps.demo full --file scenarios_obstacles_val_e.json --scenario validatione_017` | success 23.5 s, 4 mm |
| Target directly BEHIND: turns around, then goes | `$P -m comp0245.apps.demo full --file scenarios_obstacles_val_e.json --scenario validatione_005` | success 27.8 s, 8 mm |
| Three obstacles, target behind | `$P -m comp0245.apps.demo full --file scenarios_obstacles_val_e.json --scenario validatione_012` | success 25.2 s, 5 mm |

## 2. Hard layouts (walls, blockers, tight gaps, crowded parking)
| What it shows | Command | Outcome |
|---|---|---|
| Four obstacles | `$P -m comp0245.apps.demo full --file scenarios_obstacles_hard3.json --scenario validationhard3_018` | success 28.7 s, 5 mm |
| Target behind, three obstacles | `$P -m comp0245.apps.demo full --file scenarios_obstacles_hard3.json --scenario validationhard3_003` | success 27.1 s, 9 mm |

## 3. Arm generalisation (held-out episodes, never trained on)
| What it shows | Command | Outcome |
|---|---|---|
| Very low target (0.24 m) | `$P -m comp0245.apps.demo full --file scenarios_obstacles_v4_height.json --scenario trainh_000` | success 26.6 s, 6 mm |
| High target (0.95 m), behind | `$P -m comp0245.apps.demo full --file scenarios_obstacles_v4_height.json --scenario trainh_130` | success 35.2 s, 10 mm |
| Arm starts in a random pose | `$P -m comp0245.apps.demo full --file scenarios_obstacles_v4_armstart.json --scenario traina_160` | success 25.3 s, 7 mm |

## 4. Robustness (adversities)
| What it shows | Command | Outcome |
|---|---|---|
| Heavy lidar noise (0.3 m + 30% dropouts) | `$P -m comp0245.apps.demo full --file scenarios_obstacles_val_e.json --scenario validatione_012 --lidar-noise 0.3 --lidar-dropout 0.3` | success 24.0 s, 6 mm |
| 120 N sideways push at 6 s | `$P -m comp0245.apps.demo full --file scenarios_obstacles_val_e.json --scenario validatione_000 --push 120` | success 21.9 s, 6 mm |
| Wheel commands 400 ms late | `$P -m comp0245.apps.demo full --file scenarios_obstacles_val_e.json --scenario validatione_004 --delay 400` | success 28.8 s, 5 mm |

## 5. Honest failures (limitations; each runs 70 s)
| What it shows | Command | Outcome |
|---|---|---|
| Parks well, arm gets within 11 mm but does not hold (what model B targets) | `$P -m comp0245.apps.demo full --file scenarios_obstacles_val_e.json --scenario validatione_033` | timeout, no contact |
| Hard layout: base collides early | `$P -m comp0245.apps.demo full --file scenarios_obstacles_hard3.json --scenario validationhard3_012` | timeout, collision |

## Numbers to quote (validation; test split untouched)
- Fully learned robot: 37/40 on fresh normal layouts (expert 40/40); lidar noise 38/40, 400 ms delay 38/40,
  120 N push 35/40 (expert also 35/40), hard layouts 22/30 (all collisions while driving, never while reaching).
- Both networks: about 7k parameters and 30 kB each; a whole control step takes ~0.12 ms on one CPU thread (1 ms budget).
- Known weakness: targets at 0.9-1.1 m height (the arm misses most of those), and the base on hard layouts.
