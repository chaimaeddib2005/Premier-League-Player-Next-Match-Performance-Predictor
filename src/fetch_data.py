"""
Fetch (or refresh) the latest open Premier League / FPL dataset.

Source: https://github.com/vaastav/Fantasy-Premier-League
This repository is updated throughout every season (typically within a day or
two of each gameweek), so re-running this script is how the pipeline always
trains and predicts on the most current data available -- there is nothing
frozen or hardcoded to a particular gameweek or season.

Usage:
    python -m src.fetch_data
"""
import subprocess
import sys
from src.config import DATA_RAW_DIR, FPL_DATA_REPO


def main():
    if DATA_RAW_DIR.exists() and (DATA_RAW_DIR / ".git").exists():
        print(f"Existing clone found at {DATA_RAW_DIR} -- pulling latest changes...")
        subprocess.run(["git", "-C", str(DATA_RAW_DIR), "pull", "--ff-only"], check=True)
    else:
        print(f"Cloning {FPL_DATA_REPO} into {DATA_RAW_DIR} ...")
        DATA_RAW_DIR.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--depth", "1", FPL_DATA_REPO, str(DATA_RAW_DIR)],
            check=True,
        )
    print("Data is up to date.")


if __name__ == "__main__":
    sys.exit(main())
