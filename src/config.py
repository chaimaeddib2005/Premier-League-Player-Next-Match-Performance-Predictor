"""Central configuration for the PL Next-Match Predictor pipeline."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = ROOT / "data" / "fpl_data"          # cloned source repo lives here
PROCESSED_DIR = ROOT / "data" / "processed"          # combined parquet files
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"

FPL_DATA_REPO = "https://github.com/vaastav/Fantasy-Premier-League.git"

# Only seasons with expected-goals / starts data give us the richest feature set.
# This is intentionally NOT hardcoded to a fixed list at run time -- see
# discover_seasons() in build_dataset.py, which scans whatever season folders
# actually exist after fetching. This constant is only a minimum-quality filter.
MIN_USABLE_SEASON = "2022-23"

PLAYER_STATS = [
    "minutes", "starts", "goals_scored", "assists", "clean_sheets",
    "goals_conceded", "saves", "bonus", "bps", "influence", "creativity",
    "threat", "ict_index", "expected_goals", "expected_assists",
    "expected_goal_involvements", "expected_goals_conceded", "total_points",
]
ROLLING_WINDOWS = [3, 5, 10]
TEAM_ROLLING_WINDOWS = [5, 10]

for d in [DATA_RAW_DIR.parent, PROCESSED_DIR, MODELS_DIR, OUTPUTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)
