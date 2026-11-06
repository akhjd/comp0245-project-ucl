#!/bin/bash
# 5-6 Nov combined overnight run (AGENT_LOG 5 Nov evening): one base model for our rooms AND Julia's single-target
# rooms. Teacher: FastWindowExpert (window_fast, 0.8 m/s, variant F defaults, arm folded, unfolds before reaching).
# Lidar 144 rays, 10 m. Network: conv ray filter + threat head, 3 scans, no velocity, 128 units, "target blocked?"
# inputs, dropout 0.2 + weight decay 0.01, every room type weighted equally, rows 50 ms apart. No A* / map at run time.
# Finished steps are skipped, so it can be restarted. No wall-clock time limits (runs/pause.sh can freeze it).
# SMOKE=1: a ~20 min end-to-end test in runs/overnight-combined/smoke (few rooms, 1 DAgger round).
ROOT=/home/akhjd/ros2_ws/comp0245_project
cd $ROOT
PY=/home/akhjd/ros2_ws/RoboEnv/.pixi/envs/default/bin/python
W=${WORKERS:-12}
if [ -n "$SMOKE" ]; then
    O=runs/overnight-combined/smoke; NAME=overnight-smoke; PER=3; ROUNDS=1; PER_ROUND=8; VAL_LIMIT=6
else
    O=runs/overnight-combined; NAME=overnight-combined; PER=${PER:-300}; ROUNDS=3; PER_ROUND=200; VAL_LIMIT=0
fi
LIDAR="--lidar-rays 144 --lidar-range 10"
TYPES="ovrmv ovrmx ovc2 ovc3 ovc4 ovd3 ovd4"   # (not GROUPS: a bash built-in)
mkdir -p $O/rooms $O/data
echo "START $(date)"

# 1. Rooms: equal numbers of each type (make_rooms.py), crossing-mover scenes (ovx), and the validation rooms
#    (our 60 + Julia's 60 single-target ones) with the same teacher settings
[ -f $O/rooms/ovx.json ] || $PY -B runs/overnight-combined/make_rooms.py $O/rooms $PER 0.8 || exit 1
[ -f $O/rooms/val_all.json ] || $PY -B - $O/rooms/val_all.json $VAL_LIMIT <<'PY' || exit 1
import json, sys
FOLD = [3.14159, 2.618, 3.14159, 0.524, 0.0, 1.309, 0.0]
teacher = {"expert": "window_fast", "top_speed": 0.8, "expert_unfold": True, "arm_q0": FOLD}
ours = [s for f in ("scenarios_rooms_train.json", "scenarios_rooms_val.json") for s in json.load(open(f)) if s["split"] == "validation"]
hers = json.load(open("runs/julia-check/val_case234_fold.json"))
limit = int(sys.argv[2])
rooms = (ours[:limit // 2] + hers[:limit // 2]) if limit else ours + hers
json.dump([{**s, **teacher} for s in rooms], open(sys.argv[1], "x"), indent=1)
print(len(rooms), "validation rooms")
PY
echo "ROOMS READY $(date)"

# 2. The teacher on every training room type and on the validation rooms (one folder each; .done marks a finished one)
# (6 Nov: /data filled up during ovd3 at 01:51. Its finished runs stay; only the rooms missing from its manifest are
#  recorded, into data/ovd3b, which joins the training data below.)
if [ -z "$SMOKE" ] && [ ! -f $O/data/ovd3.done ] && [ -f $O/data/ovd3/manifest.json ]; then
    [ -f $O/rooms/ovd3b.json ] || $PY -B - $O <<'PY' || exit 1
import json, sys
o = sys.argv[1]
done = {e["path"] for e in json.load(open(f"{o}/data/ovd3/manifest.json"))["episodes"]}
rest = [s for s in json.load(open(f"{o}/rooms/ovd3.json")) if s["id"] not in done]
json.dump(rest, open(f"{o}/rooms/ovd3b.json", "x"), indent=1)
print(len(rest), "ovd3 rooms still to record")
PY
    [ -d $O/data/ovd3b ] && mv $O/data/ovd3b $O/data/ovd3b.partial-$(date +%H%M%S)
    nice -n 5 $PY -B collect_obstacles.py --scenarios $O/rooms/ovd3b.json --out $O/data/ovd3b $LIDAR --duration 110 \
        --workers $W > $O/data/ovd3b.log 2>&1 && touch $O/data/ovd3b.done $O/data/ovd3.done
    echo "EXPERT ovd3 (rest) FINISHED $(date): $(grep -a 'successful' $O/data/ovd3b.log | tail -1)"
fi
for G in $TYPES val_all; do
    [ -f $O/data/$G.done ] && continue
    [ -d $O/data/$G ] && mv $O/data/$G $O/data/$G.partial-$(date +%H%M%S)      # an interrupted recording: keep it aside
    nice -n 5 $PY -B collect_obstacles.py --scenarios $O/rooms/$G.json --out $O/data/$G $LIDAR --duration 110 \
        --workers $W > $O/data/$G.log 2>&1 && touch $O/data/$G.done
    echo "EXPERT $G FINISHED $(date): $(grep -a 'successful' $O/data/$G.log | tail -1)"
done

# 3. Baseline: the current room model (its own 72-ray 4 m lidar) base-only on Julia's validation rooms, arm folded, 70 s
[ -f $O/baseline-hers/evaluation.json ] || $PY -B runs/overnight-combined/baseline.py $O/baseline-hers $W $((VAL_LIMIT / 2))
echo "BASELINE FINISHED $(date)"

# 4. DAgger: behaviour cloning on all the teacher data (round 0), then rounds of the model driving every room type
#    (half of each round in the crossing-mover scenes), labelled by the teacher; only the last round is validated
DATA=$(for G in $TYPES; do echo -n "$O/data/$G "; done)
[ -f $O/data/ovd3b.done ] && DATA="$DATA $O/data/ovd3b"
SCEN=$(for G in $TYPES; do echo -n "$O/rooms/$G.json "; done)
[ -f runs/dagger-$NAME/r$ROUNDS/model.pt ] || nice -n 5 $PY -B dagger.py --name $NAME --seed 0 --rounds $ROUNDS \
    --data $DATA --validation-data $O/data/val_all --lidar-rays 144 \
    --scenarios $SCEN $O/rooms/val_all.json --label-expert window_fast \
    --hard-scenarios $O/rooms/ovx.json --hard-duration 70 --per-round $PER_ROUND --duration 70 \
    --no-velocity --scans 3 --encoder conv --threat --arm-fold --hidden 128 --workers $W --patience 8 --validate-last \
    --balance-groups --stride-ms 50 --target-ray --dropout 0.2 --weight-decay 0.01 > runs/dagger-$NAME.log 2>&1
echo "DAGGER FINISHED $(date): $(grep -a '^ROUND' runs/dagger-$NAME.log | tail -1 | cut -c1-200)"

# 5. Score the last round with the measured-reach parked rule, split into our rooms and Julia's
V=runs/dagger-$NAME/r$ROUNDS-validation
[ -f $V/rescored-reach.json ] || $PY -B rescore.py --as rescored-reach.json $V > /dev/null 2>&1
$PY -B - $V $O/baseline-hers <<'PY'
import json, sys
from pathlib import Path
from compare_scans import contacts
v = Path(sys.argv[1])
eps = json.load(open(v/"rescored-reach.json"))["episodes"]
ours = [e for e in eps if e["id"].startswith("validation")]
hers = [e for e in eps if e["id"].startswith("case")]
man = {e["path"]: e for e in json.load(open(v/"manifest.json"))["episodes"]}
def hits(group):
    return sum(any(h and m for h, m in zip(man[e["id"]].get("contact_steps_per_obstacle", []),
                                           man[e["id"]].get("obstacle_is_moving", []))) for e in group)
base = json.load(open(Path(sys.argv[2])/"evaluation.json"))["totals"]
print(f"RESULT ours: parked {sum(e['parked'] for e in ours)}/{len(ours)}, mover hits {hits(ours)} (current model: 37/60, 12)")
print(f"RESULT hers: parked {sum(e['parked'] for e in hers)}/{len(hers)}, mover hits {hits(hers)} (current model: {base['parked']}/{base['episodes']})")
PY
echo "OVERNIGHT FINISHED $(date)"
