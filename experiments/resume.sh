#!/bin/bash
# Un-freeze the jobs runs/pause.sh froze (all processes of this project's jobs and their workers).
JOBS='runs/teacher-speed/run.py|runs/overnight-combined/|dagger.py|collect_obstacles.py|eval_arm.py|train_base_torch.py|train_arm.py|runs/arm-parked/|runs/base-tweaks/|quick_rooms.py|holdcheck.py|fast_checks.py|variants.py'
tree() { echo "$1"; for c in $(pgrep -P "$1"); do tree "$c"; done; }
pids=$(for p in $(pgrep -f "^[^ ]*python[^ ]* .*($JOBS)"); do tree "$p"; done | sort -u)
[ -z "$pids" ] && { echo "nothing to resume"; exit 0; }
kill -CONT $pids
echo "resumed $(echo $pids | wc -w) processes at $(date '+%H:%M:%S')"
