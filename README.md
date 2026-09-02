# Premier League Player Next-Match Performance Predictor

Predicts a Premier League player's **next-match performance** (BPS score) and
**fantasy football points** using only information available before kickoff:
the player's recent form, their team's recent form, and the upcoming
opponent's recent attacking/defensive form and squad strength.

The project is designed to be **re-run at any point in a season**: it always
fetches the latest data, trains on everything that has actually been played
so far, and predicts whichever gameweek you ask for — including gameweeks
further ahead than the very next one (see [Predicting further ahead](#predicting-further-ahead)).

## Data

Source: the open [vaastav/Fantasy-Premier-League](https://github.com/vaastav/Fantasy-Premier-League)
dataset, which mirrors the official FPL API and is updated throughout every
season. Nothing is bundled or frozen in this repo — `fetch_data.py` clones
(or pulls) it fresh every time you run the pipeline.

## Setup

```bash
git clone <this-repo-url>
cd pl-predictor
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Three steps, run in order:

```bash
# 1. Fetch the latest data (safe to re-run any time — it just pulls updates)
python -m src.fetch_data

# 2. Build the combined, feature-ready dataset from whatever seasons/gameweeks
#    are currently available
python -m src.build_dataset

# 3. Train the models on all completed matches so far, and evaluate on the
#    most recent full season as a held-out future-season test
python -m src.train

# 4. Predict the next match for every current player
python -m src.predict
```

## Web app

Once you've run steps 1-3 above (data fetched, dataset built, models
trained), you can launch an interactive web app instead of the CLI:

```bash
python app.py
```

Then open **http://127.0.0.1:5000**. The app lets you pick any season and
gameweek (including gameweeks further ahead than the very next one, the same
way `--gameweek` works on the CLI — see [Predicting further ahead](#predicting-further-ahead)),
search and filter players, and see each prediction's fantasy points,
performance score, and a short explanation of the top factors behind it.

Predictions are computed on demand via `src.predict.generate_predictions()`
(the same function the CLI uses) and cached in memory per season/gameweek
for the lifetime of the server process, so repeated requests for the same
gameweek don't get recomputed from scratch.

`app.py` routes:

| Route | Description |
|---|---|
| `GET /` | the interactive UI |
| `GET /api/meta?season=<season>` | available seasons + each gameweek's played/unplayed status |
| `GET /api/predictions?season=<season>&gameweek=<n>` | predictions for that season/gameweek (defaults to the latest season's next unplayed gameweek) |

`predict.py` auto-detects the next gameweek that hasn't been played yet (via
the `finished` flag in the current season's `fixtures.csv`) and writes:

```
outputs/predictions_<season>_gw<N>.csv
outputs/predictions_<season>_gw<N>.json
```

### Predicting a specific gameweek

```bash
python -m src.predict --gameweek 5
python -m src.predict --season 2026-27 --gameweek 5
```

### Predicting further ahead

You don't have to wait for gameweek 3 to be played to see a gameweek 4
prediction. Each player's "current form" snapshot is built from **all of
their matches played so far** (whatever their latest completed match is) —
it's independent of how far away the target fixture is. So today, before GW3
is played, both of these work and use the *same* underlying player-form
snapshot, just paired with a different gameweek's fixture list (opponent,
home/away):

```bash
python -m src.predict --gameweek 3   # next match, using form as of end of GW2
python -m src.predict --gameweek 4   # the one after that, same form snapshot,
                                      # paired with GW4's actual fixtures
```

Predictions naturally get less certain the further ahead you ask (more can
change — injuries, rotation, transfers — between now and a distant fixture),
but the pipeline itself places no restriction on how far out you go.

## How predictions stay current

- **No hardcoded season list.** `build_dataset.py` scans whichever season
  folders exist after fetching and uses all of them.
- **No hardcoded gameweek.** `predict.py` reads the `finished` flag directly
  from the current season's fixture list.
- **Stable player identity across seasons.** FPL reassigns each player's
  numeric `element` id every season, and player names are sometimes
  formatted differently between files (e.g. `"David Raya"` vs.
  `"David Raya Martín"`). We instead key every player by FPL's permanent
  `code` field (built via a per-season `element -> code` crosswalk from
  `players_raw.csv`), so a player's full multi-season history is always
  correctly linked to their current squad entry.
- **Retrained from scratch each run.** `train.py` doesn't reuse a stale
  model file — it rebuilds features and refits on everything currently
  available, so predictions reflect the latest completed matches every time.

## Project structure

```
src/
  config.py         paths & constants
  fetch_data.py      clone/pull the latest open dataset
  build_dataset.py   combine seasons into one long-format table (stable player IDs)
  features.py        shared feature engineering (training vs. live-inference modes)
  train.py            train + evaluate + save production models
  predict.py          generate next-match predictions for any gameweek (CLI + importable function)
app.py                Flask web app (UI + JSON API) built on top of src.predict
templates/index.html  web app frontend
models/               saved model artifacts (gitignored, regenerated by train.py)
outputs/              prediction CSV/JSON + metrics (gitignored, regenerated)
data/                 fetched raw data + processed parquet (gitignored, regenerated)
```

## Methodology summary

- **Targets:** BPS score (official position-agnostic match-performance
  composite) and FPL total fantasy points.
- **Features:** rolling player form (last 3/5/10 matches + career averages),
  own-team rolling goals for/against, opponent rolling goals for/against,
  opponent squad-strength ratings, home/away, position.
- **No leakage:** every rolling feature is shifted by one match before the
  window is computed, so a match's own outcome never enters its own feature
  vector.
- **Model:** Random Forest (selected after comparing Linear Regression,
  Random Forest, and XGBoost across a feature-group ablation study — see the
  project report for full results).
- **Evaluation:** trained on all seasons before the most recent complete one,
  tested on that held-out season — a genuine future-season generalisation
  test, not a random shuffle.

## Requirements

- Python 3.9+
- `git` available on PATH (used by `fetch_data.py`)
- See `requirements.txt` for Python packages
