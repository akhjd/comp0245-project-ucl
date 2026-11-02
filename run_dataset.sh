#!/usr/bin/env bash
# Collect the expert dataset: bash run_dataset.sh SCENARIOS_FOLDER [JOBS]. Safe to rerun: finished episodes are kept.
# Only successful, contact-free runs keep their trajectory. A run still unfinished after 120 s ends as a timeout and is dropped.
set -u
cd "$(dirname "$0")"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
folder="$1"; jobs="${2:-7}"
for case in case1 case2 case3 case4 case5; do
  start=$(date +%s)
  for ((j = 0; j < jobs; j++)); do
    python collect_obstacles.py --scenarios "$folder/$case.part$j.json" --out "$folder/$case/part$j" --rrt --duration 120 \
      --save-every 10 --resume --only-clean >> "$folder/$case.part$j.log" 2>&1 &
  done
  wait
  echo "$case finished in $(( $(date +%s) - start )) s"
done
