"""
Train the next-match performance & fantasy-points models on ALL completed
matches currently available (whatever seasons/gameweeks have been played by
the time this is run), then refit production models on the full history.

No gameweek number is hardcoded anywhere in this script: "completed" simply
means the match has already been played (a real total_points value exists in
the source data), which is naturally true for more matches every time this
script is re-run later in a season or in a future season.

Usage:
    python -m src.train
"""
import sys
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import PROCESSED_DIR, MODELS_DIR, OUTPUTS_DIR
from src.features import (
    add_player_rolling_features, compute_team_match_goals, add_team_rolling_form,
    feature_columns,
)


def build_training_table():
    raw = pd.read_parquet(PROCESSED_DIR / "player_match_raw.parquet")
    raw["kickoff_time"] = pd.to_datetime(raw["kickoff_time"])

    df = add_player_rolling_features(raw, shift=True)

    team_goals = compute_team_match_goals(raw)
    team_goals = add_team_rolling_form(team_goals, shift=True)
    df = df.merge(
        team_goals[["team", "kickoff_time", "team_goals_for_r5", "team_goals_against_r5",
                     "team_goals_for_r10", "team_goals_against_r10"]],
        on=["team", "kickoff_time"], how="left",
    )

    opp_form = team_goals.rename(columns={
        "team": "opponent_team_name",
        "team_goals_for_r5": "opp_goals_for_r5", "team_goals_against_r5": "opp_goals_against_r5",
        "team_goals_for_r10": "opp_goals_for_r10", "team_goals_against_r10": "opp_goals_against_r10",
    })[["opponent_team_name", "kickoff_time", "opp_goals_for_r5", "opp_goals_against_r5",
        "opp_goals_for_r10", "opp_goals_against_r10"]]
    df = df.merge(opp_form, on=["opponent_team_name", "kickoff_time"], how="left")

    df["was_home"] = df["was_home"].astype(float)
    df = pd.concat([df, pd.get_dummies(df["position"], prefix="pos")], axis=1)

    for c in ["opp_strength", "opp_strength_attack_home", "opp_strength_attack_away",
              "opp_strength_defence_home", "opp_strength_defence_away"]:
        df[c] = df[c].fillna(df[c].median())

    df["target_performance"] = df["bps"]
    df["target_fantasy_points"] = df["total_points"]

    # only rows that are (a) an actually-completed match and (b) have at least
    # one prior match on record (nothing to build a feature vector from otherwise)
    df = df.dropna(subset=["total_points"])
    df = df[df["p_n_prior_matches"] >= 1].copy()
    return df


def time_based_holdout_eval(df, feats):
    """Evaluate on the most recently completed FULL season as a genuine
    future-generalisation test, training on every earlier season."""
    seasons_sorted = sorted(df["season"].unique())
    if len(seasons_sorted) < 2:
        print("Only one season of data available -- skipping holdout evaluation "
              "(not enough history for a fair future-season test yet).")
        return
    test_season = seasons_sorted[-2] if seasons_sorted[-1] == seasons_sorted[-1] else seasons_sorted[-1]
    # use the second-to-last season as the test set when the last season is still
    # in progress (few rows); otherwise use the last complete season
    last_season_rows = (df["season"] == seasons_sorted[-1]).sum()
    if last_season_rows < 2000:  # in-progress season, too few matches for a fair test
        test_season = seasons_sorted[-2]
    else:
        test_season = seasons_sorted[-1]
    train_seasons = [s for s in seasons_sorted if s < test_season]
    if not train_seasons:
        print("Not enough earlier seasons to hold one out fairly -- skipping.")
        return

    train = df[df["season"].isin(train_seasons)]
    test = df[df["season"] == test_season]
    print(f"\nHoldout evaluation: train on {train_seasons}, test on {test_season} "
          f"({len(train)} train rows, {len(test)} test rows)")

    rows = []
    for target_label, target_col in [("performance_bps", "target_performance"),
                                       ("fantasy_points", "target_fantasy_points")]:
        y_train, y_test = train[target_col].fillna(0), test[target_col].fillna(0)
        model = RandomForestRegressor(n_estimators=100, max_depth=8, min_samples_leaf=8,
                                       n_jobs=-1, random_state=42, max_samples=0.5)
        model.fit(train[feats].fillna(0), y_train)
        preds = model.predict(test[feats].fillna(0))
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        print(f"  {target_label:16s}  MAE={mae:.3f}  RMSE={rmse:.3f}  R2={r2:.3f}")
        rows.append({"target": target_label, "test_season": test_season,
                      "MAE": round(mae, 3), "RMSE": round(rmse, 3), "R2": round(r2, 3)})
    pd.DataFrame(rows).to_csv(OUTPUTS_DIR / "holdout_metrics.csv", index=False)


def main():
    df = build_training_table()
    feats = feature_columns(df.columns)
    print(f"{len(df)} training rows, {len(feats)} features, "
          f"seasons: {sorted(df['season'].unique())}")

    time_based_holdout_eval(df, feats)

    # final production fit on ALL completed matches available right now
    X = df[feats].fillna(0)
    for target_label, target_col in [("performance_bps", "target_performance"),
                                       ("fantasy_points", "target_fantasy_points")]:
        y = df[target_col].fillna(0)
        model = RandomForestRegressor(n_estimators=80, max_depth=8, min_samples_leaf=8,
                                       n_jobs=-1, random_state=42, max_samples=0.35)
        model.fit(X, y)
        joblib.dump(model, MODELS_DIR / f"model_{target_label}.joblib")

        importances = pd.Series(model.feature_importances_, index=feats).sort_values(ascending=False)
        importances.to_csv(OUTPUTS_DIR / f"feature_importance_{target_label}.csv")

    joblib.dump(feats, MODELS_DIR / "model_features.joblib")
    print(f"\nProduction models trained on {len(df)} rows and saved to {MODELS_DIR}")


if __name__ == "__main__":
    sys.exit(main())
