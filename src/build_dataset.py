"""
Combine every available season into one long-format player-match table, with
a STABLE player identity that survives across seasons and name-spelling
differences.

Why this matters: FPL's per-season "element" id is reassigned every season,
and the "name" field in the gameweek files is sometimes formatted differently
from the current squad list (accents, short vs. full names, e.g.
"David Raya" vs "David Raya Martín"). Matching on either of those directly
causes real players to silently disappear from predictions.

FPL players carry a permanent numeric "code" field, present in every season's
players_raw.csv, that never changes for a given real person -- it's what FPL
itself uses to serve a player's photo regardless of season. We build a
per-season (element -> code) crosswalk from players_raw.csv and attach it to
every match row, so downstream joins (rolling history <-> current squad) use
`player_code`, not name strings.

Usage:
    python -m src.build_dataset
"""
import sys
import pandas as pd
import numpy as np

from src.config import DATA_RAW_DIR, PROCESSED_DIR, MIN_USABLE_SEASON

KEEP_COLS = [
    "name", "position", "team", "GW", "kickoff_time", "opponent_team",
    "was_home", "minutes", "starts", "goals_scored", "assists",
    "clean_sheets", "goals_conceded", "own_goals", "penalties_missed",
    "penalties_saved", "saves", "yellow_cards", "red_cards", "bonus", "bps",
    "influence", "creativity", "threat", "ict_index",
    "expected_goals", "expected_assists", "expected_goal_involvements",
    "expected_goals_conceded", "total_points", "value", "selected", "element",
]


def discover_seasons():
    """Scan the fetched data directory for season folders that actually
    contain usable gameweek data, instead of hardcoding a season list."""
    data_dir = DATA_RAW_DIR / "data"
    seasons = []
    for p in sorted(data_dir.iterdir()):
        if not p.is_dir() or not p.name[:4].isdigit():
            continue
        if p.name < MIN_USABLE_SEASON:
            continue
        gw_file = p / "gws" / "merged_gw.csv"
        if gw_file.exists():
            seasons.append(p.name)
    return seasons


def load_season(season):
    season_dir = DATA_RAW_DIR / "data" / season
    df = pd.read_csv(season_dir / "gws" / "merged_gw.csv", encoding="latin-1")

    for c in KEEP_COLS:
        if c not in df.columns:
            df[c] = np.nan
    df = df[KEEP_COLS].copy()
    df["season"] = season

    teams = pd.read_csv(season_dir / "teams.csv")
    id2name = dict(zip(teams["id"], teams["name"]))
    df["opponent_team_name"] = df["opponent_team"].map(id2name)

    strength_cols = ["id", "strength", "strength_attack_home", "strength_attack_away",
                      "strength_defence_home", "strength_defence_away"]
    strength = teams[strength_cols].rename(columns={"id": "opponent_team"})
    strength = strength.add_prefix("opp_").rename(columns={"opp_opponent_team": "opponent_team"})
    df = df.merge(strength, on="opponent_team", how="left")

    # stable cross-season identity: element (this season's id) -> permanent code
    players_raw = pd.read_csv(season_dir / "players_raw.csv")
    element2code = dict(zip(players_raw["id"], players_raw["code"]))
    df["player_code"] = df["element"].map(element2code)
    # drop the handful of rows (usually managers / data artifacts) with no crosswalk match
    df = df[df["player_code"].notna()]
    df["player_code"] = df["player_code"].astype(int)

    # remove non-outfield-position artifacts (e.g. manager rows tagged "AM")
    df = df[df["position"].isin(["GK", "DEF", "MID", "FWD"])]

    return df


def main():
    seasons = discover_seasons()
    if not seasons:
        print("No usable season data found. Run `python -m src.fetch_data` first.")
        return 1
    print("Seasons discovered:", seasons)

    frames = [load_season(s) for s in seasons]
    all_df = pd.concat(frames, ignore_index=True)
    all_df["kickoff_time"] = pd.to_datetime(all_df["kickoff_time"])
    all_df = all_df.sort_values(["player_code", "kickoff_time"]).reset_index(drop=True)

    print("Total player-match rows:", len(all_df))
    print("Unique players (by permanent code):", all_df["player_code"].nunique())
    print("Date range:", all_df["kickoff_time"].min(), "->", all_df["kickoff_time"].max())

    out_path = PROCESSED_DIR / "player_match_raw.parquet"
    all_df.to_parquet(out_path, index=False)
    print("Saved", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
