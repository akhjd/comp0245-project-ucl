#!/bin/bash
# Freeze every running job of this project (expert runs, data collection, DAgger, training) for a demo; resume with
# runs/resume.sh. Frozen processes use no CPU and continue exactly where they stopped (simulated time does not move
# while frozen). They keep their memory. Julia's interactive tool, drive_mlp.py and demo.py are NOT touched.
JOBS='runs/teacher-speed/run.py|runs/overnight-combined/|dagger.py|collect_obstacles.py|eval_arm.py|train_base_torch.py|train_arm.py|runs/arm-parked/|runs/base-tweaks/|quick_rooms.py|holdcheck.py|fast_checks.py|variants.py'
tree() { echo "$1"; for c in $(pgrep -P "$1"); do tree "$c"; done; }
pids=$(for p in $(pgrep -f "^[^ ]*python[^ ]* .*($JOBS)"); do tree "$p"; done | sort -u)
[ -z "$pids" ] && { echo "nothing running to pause"; exit 0; }
kill -STOP $pids
echo "paused $(echo $pids | wc -w) processes at $(date '+%H:%M:%S'):"
pgrep -af "^[^ ]*python[^ ]* .*($JOBS)" | cut -c1-110
echo "resume with: runs/resume.sh"
