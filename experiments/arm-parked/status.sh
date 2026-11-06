#!/bin/bash
# Status of the arm work in runs/arm-parked (5 Nov). Read-only. Use: watch -n 15 runs/arm-parked/status.sh
cd /home/akhjd/ros2_ws/comp0245_project
A=runs/arm-parked
count() { ls -d "$1"/*/ 2>/dev/null | grep -c . ; }
score() { python3 -c "import json;r=json.load(open('$1/results.json'));print(f\"{r['success']}/{r['spots']}\")" 2>/dev/null; }
echo "Arm DAgger on parked spots  $(date '+%a %H:%M:%S')"
echo
echo "Room arm before DAgger (runs/arm-parked/arm):"
echo "   49 validation spots: $(score $A/test-new)   (old arm_final: $(score $A/test-current))"
if [ -f $A/test-new-synth200/results.json ]; then echo "   200 synthetic spots: $(score $A/test-new-synth200)"
elif [ -d $A/test-new-synth200 ]; then echo "   200 synthetic spots: running ($(count $A/test-new-synth200)/200 started)"
else echo "   200 synthetic spots: not started"; fi
for k in 1 2; do
    D=$A/dagger-r$k
    echo
    if [ ! -d $D ]; then echo "Round $k: not started"; continue; fi
    echo "Round $k:"
    [ -d $D/data ] && echo "   driving with expert labels: $(count $D/data)/500 started$([ -f $D/data/manifest.json ] && echo ' (done)')"
    if [ -f $D/arm/model.pt ]; then echo "   training: done ($(grep -o 'best epoch [0-9]*' $D/train.log))"
    elif [ -f $D/train.log ]; then echo "   training: $(grep -a '^epoch' $D/train.log | tail -1) (of 60)"; fi
    for t in test-val49 test-synth200; do
        if [ -f $D/$t/results.json ]; then echo "   $t: $(score $D/$t)"
        elif [ -d $D/$t ]; then echo "   $t: running ($(count $D/$t) started)"; fi
    done
done
F=runs/arm-eval-rooms-full-val-dagger2
echo
if [ -f $F/results.json ]; then echo "Full robot with the round 2 arm: $(python3 -c "import json;E=json.load(open('$F/results.json'))['episodes'];print(sum(e['success'] for e in E),'/',len(E))") (room arm before DAgger: 37/60)"
elif [ -d $F ]; then echo "Full robot with the round 2 arm: running ($(ls -d $F/fresh-normal/*/ 2>/dev/null | wc -l)/60 started)"; fi
echo
echo "Rule: >= 41/49 on the validation spots, not worse on the 200 synthetic, then full robot >= 37/60"
echo
free -h | head -2
echo
echo "Last lines of dagger.log:"; tail -3 $A/dagger.log 2>/dev/null | cut -c1-120
