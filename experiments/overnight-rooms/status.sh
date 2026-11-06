#!/bin/bash
# Status of the 5 Nov overnight run (runs/overnight-rooms/run.sh). Read-only. Use: watch -n 30 runs/overnight-rooms/status.sh
cd /home/akhjd/ros2_ws/comp0245_project
count() { ls -d "$1"/*/ 2>/dev/null | grep -c . ; }
echo "Overnight run status  $(date '+%a %H:%M:%S')"
echo
if ! grep -q "ALL EDITS DONE" runs/chatgpt-edits/run_all.log 2>/dev/null; then
    echo "Waiting for the ChatGPT-edit queue:"; tail -2 runs/chatgpt-edits/run_all.log | cut -c1-110
    for n in combined balanced; do [ -d runs/chatgpt-edits/$n-validation ] && echo "  $n: $(count runs/chatgpt-edits/$n-validation)/60 rooms started"; done
    echo
fi
E=runs/rooms-window72-new
if [ -d $E ]; then
    if grep -q "EXPERT DATA FINISHED" runs/overnight-rooms/run.log 2>/dev/null; then echo "1. Expert on 600 new rooms: DONE"
    else echo "1. Expert on 600 new rooms: $(count $E)/600 started (manifest.json is written as it goes)"; fi
else echo "1. Expert on 600 new rooms: not started"; fi
for S in 0 1; do
    D=runs/dagger-rooms-big-s$S
    if [ ! -d $D ]; then echo "$((S+2)). Seed $S: not started"; continue; fi
    echo "$((S+2)). Seed $S:"
    for k in 1 2 3; do
        line="   round $k:"
        [ -d $D/r$k-data ] && line="$line driving $(count $D/r$k-data)/50"
        [ -d $D/r$k-hard-data ] && line="$line + crossing $(count $D/r$k-hard-data)/50"
        [ -f $D/r$k/model.pt ] && line="$line | trained ($(grep -o '"epochs_run": [0-9]*' $D/r$k/training.json | grep -o '[0-9]*$') epochs)"
        [ -d $D/r$k-validation ] && line="$line | validation $(count $D/r$k-validation)/60"
        echo "$line"
    done
    L=runs/dagger-rooms-big-s$S.log
    ep=$(grep -a "^epoch" $L 2>/dev/null | tail -1)
    [ -n "$ep" ] && echo "   last training line: $ep"
    grep -a "^ROUND" $L 2>/dev/null | tail -1 | cut -c1-150 | sed 's/^/   /'
done
echo
echo "Rule: both seeds > 35 / 32 parked with mover hits <= 12 / 13"
echo
free -h | head -2
echo
echo "Last lines of run.log:"; grep -av Warning runs/overnight-rooms/run.log 2>/dev/null | tail -3 | cut -c1-120
