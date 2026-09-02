"""
Flask web app for the PL Next-Match Predictor.

Run after `python -m src.fetch_data`, `python -m src.build_dataset`, and
`python -m src.train` have been run at least once (so trained models exist
in models/):

    python app.py

Then open http://127.0.0.1:5000

Routes:
    GET /                      the interactive UI
    GET /api/meta              available seasons + gameweeks (played/unplayed)
    GET /api/predictions       predictions for a season/gameweek (defaults to
                                the latest season's next unplayed gameweek)
                                query params: ?season=2026-27&gameweek=4
"""
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from src.config import MODELS_DIR
from src.predict import generate_predictions, latest_season, list_seasons, list_gameweeks

app = Flask(__name__)

# simple in-memory cache so repeated requests for the same season/gameweek
# (e.g. every visitor loading the page) don't retrain/recompute from scratch
_prediction_cache = {}


def _models_ready():
    return (MODELS_DIR / "model_fantasy_points.joblib").exists() and \
           (MODELS_DIR / "model_performance_bps.joblib").exists() and \
           (MODELS_DIR / "model_features.joblib").exists()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/meta")
def meta():
    if not _models_ready():
        return jsonify({"error": "Models not found. Run `python -m src.train` first."}), 503
    try:
        seasons = list_seasons()
        season = request.args.get("season") or latest_season()
        gameweeks = list_gameweeks(season)
        return jsonify({"seasons": seasons, "current_season": season, "gameweeks": gameweeks})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/predictions")
def predictions():
    if not _models_ready():
        return jsonify({
            "error": "Models not found. Run `python -m src.fetch_data`, "
                     "`python -m src.build_dataset`, and `python -m src.train` first."
        }), 503

    season = request.args.get("season") or None
    gameweek = request.args.get("gameweek")
    gameweek = int(gameweek) if gameweek else None

    cache_key = (season, gameweek)
    if cache_key in _prediction_cache:
        pred_df, meta_info = _prediction_cache[cache_key]
    else:
        try:
            pred_df, meta_info = generate_predictions(season=season, gameweek=gameweek)
        except Exception as e:
            return jsonify({"error": str(e)}), 400
        _prediction_cache[cache_key] = (pred_df, meta_info)

    return jsonify({"meta": meta_info, "players": pred_df.to_dict(orient="records")})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
