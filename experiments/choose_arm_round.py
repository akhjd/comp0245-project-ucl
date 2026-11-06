"""Pick the arm DAgger round with the most fully-learned successes on the 90 (tie: expert parking, then earlier).
Same rule as runs/arm-b-pipeline.sh used for the final arm. Usage: python runs/choose_arm_round.py runs/dagger-arm-NAME"""
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
rounds = json.loads((root/"results.json").read_text())["rounds"]
best = max(rounds, key=lambda r: (r["full_success"], r["success"], -r["round"]))
(root/"chosen.json").write_text(json.dumps({"round": best["round"], "rule":
    "most fully-learned successes on the 90; tie -> expert-parking successes; then the earlier round", "scores": best}, indent=1))
for r in rounds:
    print(f"r{r['round']}: fully learned {r['full_success']}/{r['episodes']}, expert parking {r['success']}/{r['episodes']}")
print(best["round"])
