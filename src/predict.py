"""
Generate next-match predictions for a target gameweek.

Key idea: a player's "current form" snapshot is always built from ALL of
their matches played so far (whatever that latest completed match happens to
be) -- it does not care how many gameweeks away the target fixture is. So:

  * If GW3 hasn't been played yet, `--gameweek 3` predicts GW3 using each
    player's form as of the end of GW2 (or whatever their last completed
    match was).
  * `--gameweek 4` works exactly the same way, TODAY, even before GW3 is
    played -- it just pairs each player's current-form snapshot with GW4's
    fixture list (opponent, home/away) instead of GW3's. The prediction will
    simply be less certain the further ahead you ask, since more may change
    (team news, injuries, rotation) between now and a more distant gameweek --
    but the pipeline itself places no restriction on how far ahead you ask.

With no --gameweek given, the script auto-detects the next gameweek in the
current season that has not been finished yet (fixtures.csv `finished` flag),
so "just run it" always gets you the next real upcoming fixtures.

Usage:
    python -m src.predict                     # next unplayed gameweek, latest season
    python -m src.predict --gameweek 4         # a specific (possibly further-out) gameweek
    python -m src.predict --season 2026-27 --gameweek 4
"""
import argparse
import sys

import joblib
import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR, MODELS_DIR, OUTPUTS_DIR, DATA_RAW_DIR
from src.features import (
    add_player_rolling_features, compute_team_match_goals, add_team_rolling_form,
)

LABEL_MAP = {
    "p_minutes_r3": "Minutes played (last 3 matches)",
    "p_minutes_r5": "Minutes played (last 5 matches)",
    "p_minutes_r10": "Minutes played (last 10 matches)",
    "p_played_last_match": "Played in most recent match",
    "p_n_prior_matches": "Career experience (matches played)",
    "p_total_points_r10": "FPL points, last 10 matches (avg)",
    "p_total_points_r5": "FPL points, last 5 matches (avg)",
    "p_total_points_r3": "FPL points, last 3 matches (avg)",
    "p_total_points_career": "Career FPL points average",
    "p_bps_r10": "Bonus Points System score, last 10 (avg)",
    "p_bps_r5": "Bonus Points System score, last 5 (avg)",
    "p_bps_r3": "Bonus Points System score, last 3 (avg)",
    "p_bps_career": "Career BPS average",
    "p_ict_index_r3": "ICT Index, last 3 matches (avg)",
    "p_ict_index_r5": "ICT Index, last 5 matches (avg)",
    "p_ict_index_r10": "ICT Index, last 10 matches (avg)",
    "p_ict_index_career": "Career ICT Index average",
    "p_expected_goal_involvements_r10": "Expected goal involvement (xG+xA), last 10",
    "p_expected_goal_involvements_r5": "Expected goal involvement (xG+xA), last 5",
    "p_threat_r10": "Attacking threat rating, last 10 (avg)",
    "p_threat_career": "Career attacking threat rating",
    "opp_strength_defence_home": "Opponent's home defensive strength",
    "opp_strength_defence_away": "Opponent's away defensive strength",
    "opp_strength_attack_home": "Opponent's home attacking strength",
    "opp_strength_attack_away": "Opponent's away attacking strength",
    "opp_goals_against_r5": "Opponent's goals conceded, last 5 (avg)",
    "opp_goals_for_r5": "Opponent's goals scored, last 5 (avg)",
    "was_home": "Home advantage",
    "team_goals_for_r5": "Own team's goals scored, last 5 (avg)",
    "team_goals_against_r5": "Own team's goals conceded, last 5 (avg)",
}


def latest_season():
    data_dir = DATA_RAW_DIR / "data"
    seasons = sorted(p.name for p in data_dir.iterdir() if p.is_dir() and p.name[:4].isdigit())
    return seasons[-1]


def list_seasons():
    data_dir = DATA_RAW_DIR / "data"
    return sorted(p.name for p in data_dir.iterdir()
                  if p.is_dir() and p.name[:4].isdigit() and (p / "fixtures.csv").exists())


def list_gameweeks(season):
    """All gameweeks in a season with a played/unplayed flag, for populating a UI selector."""
    fixtures = pd.read_csv(DATA_RAW_DIR / "data" / season / "fixtures.csv")
    g = fixtures.groupby("event")["finished"].all().reset_index()
    return [{"gameweek": int(r["event"]), "finished": bool(r["finished"])} for _, r in g.iterrows()]


def next_unplayed_gameweek(fixtures):
    unplayed = fixtures.loc[~fixtures["finished"], "event"]
    if unplayed.empty:
        raise ValueError("No unplayed gameweeks found in fixtures.csv for this season.")
    return int(unplayed.min())


def build_current_form_snapshots():
    raw = pd.read_parquet(PROCESSED_DIR / "player_match_raw.parquet")
    raw["kickoff_time"] = pd.to_datetime(raw["kickoff_time"])

    player_feats = add_player_rolling_features(raw, shift=False)
    latest_player = (
        player_feats.sort_values("kickoff_time").groupby("player_code", as_index=False).tail(1)
    ).set_index("player_code")

    team_goals = compute_team_match_goals(raw)
    team_goals = add_team_rolling_form(team_goals, shift=False)
    latest_team = (
        team_goals.sort_values("kickoff_time").groupby("team", as_index=False).tail(1)
    ).set_index("team")

    return latest_player, latest_team


def generate_predictions(season=None, gameweek=None):
    """Core prediction logic, importable for reuse (CLI, Flask app, notebooks, etc).

    Returns (pred_df, meta) where meta describes the resolved season/gameweek
    and whether that gameweek has already been played.
    """
    season = season or latest_season()
    season_dir = DATA_RAW_DIR / "data" / season
    teams = pd.read_csv(season_dir / "teams.csv")
    fixtures = pd.read_csv(season_dir / "fixtures.csv")
    players_raw = pd.read_csv(season_dir / "players_raw.csv")

    gameweek = gameweek or next_unplayed_gameweek(fixtures)
    target_fixtures = fixtures[fixtures["event"] == gameweek]
    if target_fixtures.empty:
        raise ValueError(f"No fixtures found for gameweek {gameweek} in season {season}.")
    already_played = bool(target_fixtures["finished"].all())

    id2name = dict(zip(teams["id"], teams["name"]))
    strength_map = teams.set_index("id")[
        ["strength", "strength_attack_home", "strength_attack_away",
         "strength_defence_home", "strength_defence_away"]
    ]
    fixture_map = {}
    for _, row in target_fixtures.iterrows():
        h, a = id2name[row["team_h"]], id2name[row["team_a"]]
        fixture_map[h] = (a, True)
        fixture_map[a] = (h, False)

    players_raw = players_raw.copy()
    players_raw["full_name"] = players_raw["first_name"] + " " + players_raw["second_name"]
    id2pos = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
    players_raw["position"] = players_raw["element_type"].map(id2pos)
    players_raw["team_name"] = players_raw["team"].map(id2name)

    player_form, team_form = build_current_form_snapshots()

    model_perf = joblib.load(MODELS_DIR / "model_performance_bps.joblib")
    model_fp = joblib.load(MODELS_DIR / "model_fantasy_points.joblib")
    feats = joblib.load(MODELS_DIR / "model_features.joblib")

    records, skipped_no_history = [], 0
    for _, prow in players_raw.iterrows():
        team_name = prow["team_name"]
        if team_name not in fixture_map:
            continue
        opp_name, was_home = fixture_map[team_name]

        player_code = prow["code"]
        if player_code not in player_form.index:
            skipped_no_history += 1
            continue
        pf = player_form.loc[player_code]

        row = {}
        for c in feats:
            if c.startswith("p_"):
                row[c] = pf.get(c, 0)
            elif c.startswith("pos_"):
                row[c] = 0
            elif c == "was_home":
                row[c] = float(was_home)
            elif c.startswith("team_"):
                row[c] = team_form.loc[team_name, c] if team_name in team_form.index else np.nan
            elif c.startswith("opp_"):
                base = c.replace("opp_", "")
                if base in ("goals_for_r5", "goals_against_r5", "goals_for_r10", "goals_against_r10"):
                    row[c] = team_form.loc[opp_name, f"team_{base}"] if opp_name in team_form.index else np.nan
                else:
                    opp_id = teams.loc[teams["name"] == opp_name, "id"].values[0]
                    row[c] = strength_map.loc[opp_id, base]

        pos_col = f"pos_{prow['position']}"
        if pos_col in row:
            row[pos_col] = 1

        X = pd.DataFrame([row])[feats].fillna(0)
        pred_perf = float(model_perf.predict(X)[0])
        pred_fp = float(model_fp.predict(X)[0])

        importances = model_fp.feature_importances_
        contrib = pd.Series(importances, index=feats) * X.iloc[0]
        top_factors = contrib.sort_values(ascending=False).head(4).index.tolist()

        records.append({
            "name": prow["full_name"], "team": team_name, "opponent": opp_name,
            "was_home": bool(was_home), "position": prow["position"],
            "predicted_performance_bps": round(pred_perf, 1),
            "predicted_fantasy_points": round(pred_fp, 2),
            "recent_minutes_r3": round(float(pf.get("p_minutes_r3", 0)), 1),
            "recent_points_r5": round(float(pf.get("p_total_points_r5", 0)), 2),
            "top_factors": top_factors,
            "top_factors_readable": [LABEL_MAP.get(f, f) for f in top_factors],
        })

    pred_df = pd.DataFrame(records).sort_values("predicted_fantasy_points", ascending=False)
    meta = {
        "season": season, "gameweek": int(gameweek), "already_played": already_played,
        "n_players": len(pred_df), "n_skipped_no_history": skipped_no_history,
    }
    return pred_df, meta


def main():
    parser = argparse.ArgumentParser(description="Predict next-match performance & fantasy points.")
    parser.add_argument("--season", default=None, help="Season folder, e.g. 2026-27. Defaults to the latest available.")
    parser.add_argument("--gameweek", type=int, default=None,
                         help="Target gameweek number. Defaults to the next gameweek that hasn't been played yet.")
    args = parser.parse_args()

    pred_df, meta = generate_predictions(season=args.season, gameweek=args.gameweek)
    print(f"Season: {meta['season']} | Target gameweek: {meta['gameweek']} "
          f"({'already played -- showing what the model would have predicted' if meta['already_played'] else 'upcoming'})")

    out_csv = OUTPUTS_DIR / f"predictions_{meta['season']}_gw{meta['gameweek']}.csv"
    out_json = OUTPUTS_DIR / f"predictions_{meta['season']}_gw{meta['gameweek']}.json"
    pred_df.to_csv(out_csv, index=False)
    pred_df.to_json(out_json, orient="records")

    print(f"Predicted {meta['n_players']} players ({meta['n_skipped_no_history']} skipped -- no history yet, e.g. new signings).")
    print(f"Saved: {out_csv}")
    print(f"Saved: {out_json}")
    print(pred_df.head(10)[["name", "team", "opponent", "was_home", "position",
                             "predicted_fantasy_points", "predicted_performance_bps"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
