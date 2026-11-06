"""Fixed locations inside the repository, so code works from any working directory."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"
SETTINGS_FILE = CONFIG_DIR / "settings.json"
SCENARIO_DIR = ROOT / "data" / "scenarios"
WEIGHTS_DIR = ROOT / "weights"
RUNS_DIR = ROOT / "runs"


def scenario_path(name):
    """Find a scenario file given as a path or a bare name such as "scenarios_rooms_val.json".

    Looks at the path as given, then relative to the repository root, then in data/scenarios/. Bare
    names keep working for commands and run manifests written before the files moved.
    """
    path = Path(name)
    for candidate in (path, ROOT / path, SCENARIO_DIR / path.name):
        if candidate.exists():
            return candidate
    return path
