"""
Shared feature-engineering functions.

Two modes, same underlying logic:
  * TRAINING mode (`shift=True`): every row's features are built only from
    matches strictly BEFORE that row's own match (rolling window is shifted
    by one), so a match's own outcome can never leak into its own features.
    This is what train.py uses to build a supervised dataset of
    (features -> known outcome) pairs from history.
  * INFERENCE mode (`shift=False`): features are built from ALL of a
    player/team's matches up to and including their most recent one, i.e.
    "current form right now". This is what predict.py uses to build the
    snapshot that feeds a prediction for a match that HASN'T happened yet.
    There is no leakage here either -- the target match is, by definition,
    not in the input data at all.
"""
import pandas as pd
import numpy as np

from src.config import PLAYER_STATS, ROLLING_WINDOWS, TEAM_ROLLING_WINDOWS


def add_player_rolling_features(df, shift=True):
    df = df.sort_values(["player_code", "kickoff_time"]).reset_index(drop=True)
    g = df.groupby("player_code", group_keys=False)
    s = 1 if shift else 0

    for w in ROLLING_WINDOWS:
        for stat in PLAYER_STATS:
            col = f"p_{stat}_r{w}"
            df[col] = g[stat].transform(
                lambda x, w=w, s=s: x.shift(s).rolling(w, min_periods=1).mean()
            )
    for stat in PLAYER_STATS:
        col = f"p_{stat}_career"
        df[col] = g[stat].transform(lambda x, s=s: x.shift(s).expanding(min_periods=1).mean())

    df["p_n_prior_matches"] = g.cumcount() + (0 if shift else 1)
    df["p_played_last_match"] = g["minutes"].transform(
        lambda x, s=s: (x.shift(s) > 0).astype(float) if s else (x > 0).astype(float)
    )
    return df


def compute_team_match_goals(df):
    """One row per (team, match): goals for/against, independent of any one player."""
    team_match_goals = (
        df.groupby(["team", "season", "kickoff_time"], as_index=False)
        .agg(team_goals_for=("goals_scored", "sum"), team_goals_against=("goals_conceded", "max"))
        .sort_values(["team", "kickoff_time"])
    )
    return team_match_goals


def add_team_rolling_form(team_match_goals, shift=True):
    g = team_match_goals.groupby("team", group_keys=False)
    s = 1 if shift else 0
    for w in TEAM_ROLLING_WINDOWS:
        team_match_goals[f"team_goals_for_r{w}"] = g["team_goals_for"].transform(
            lambda x, w=w, s=s: x.shift(s).rolling(w, min_periods=1).mean()
        )
        team_match_goals[f"team_goals_against_r{w}"] = g["team_goals_against"].transform(
            lambda x, w=w, s=s: x.shift(s).rolling(w, min_periods=1).mean()
        )
    return team_match_goals


def feature_columns(cols):
    return [c for c in cols if c.startswith("p_") or c.startswith("team_")
            or c.startswith("opp_") or c.startswith("pos_") or c == "was_home"]
