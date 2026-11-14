# ZoomCamp ML Midterm Project: Wildfire Severity Prediction

Midterm project for the [ML Zoomcamp](https://github.com/DataTalksClub/machine-learning-zoomcamp) course.

Given **where** and **when** a wildfire was discovered and **what caused it**, the model predicts whether it will
grow into a **large fire** (10+ acres). Pick a spot on a US map, fill in the date, hour and cause, and get a
prediction from the web app.

## Quick start

### Option 1: Docker (recommended)

Requires [Docker](https://docs.docker.com/get-docker/) only.

```bash
docker build -t wildfire-predictor .
docker run --rm -p 8000:8000 wildfire-predictor
```

Then open **<http://localhost:8000>**.

The build is self-contained: if `models/wildfire_model.pkl` is not in the repo, the build trains it with `train.py`
(about a minute on the full dataset), so there is nothing to run beforehand. The final image is ~650 MB and only
contains the app and the model.

To force a retrain, delete the model and rebuild:

```bash
rm models/wildfire_model.pkl
docker build -t wildfire-predictor .
```

### Option 2: Run locally

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/) (it installs the right Python and
dependencies for you).

```bash
uv sync                      # install dependencies
uv run python train.py       # train the model (skip if models/wildfire_model.pkl already exists)
uv run python predict.py     # start the web app
```

Then open **<http://localhost:8000>**.

Useful variations:

```bash
uv run python train.py --sample          # quick training run on the 500k-row sample
uv run uvicorn predict:app --reload      # auto-reload while developing
PORT=9000 uv run python predict.py       # use a different port
uv run jupyter lab                       # open the analysis notebook
```

## Using the web app

1. Click a state on the map. The state, latitude and longitude are filled in for you (it starts on a fire in Texas).
2. Choose the discovery date, hour (optional) and cause.
3. Hit **Predict** to see the probability of a large fire and the verdict.

## Using the API

Interactive docs are at <http://localhost:8000/docs>.

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{
        "discovery_date": "2005-07-28",
        "discovery_hour": 13,
        "latitude": 33.578,
        "longitude": -101.1182,
        "state": "TX",
        "stat_cause_descr": "Debris Burning"
      }'
```

```json
{"probability_large": 0.51, "is_large": true, "threshold": 0.2275}
```

| Field              | Notes                                                                |
|--------------------|----------------------------------------------------------------------|
| `discovery_date`   | `YYYY-MM-DD`                                                         |
| `discovery_hour`   | optional, 0-23                                                       |
| `latitude`         | 17 to 72                                                             |
| `longitude`        | -180 to -60                                                          |
| `state`            | two-letter code (50 states, `DC`, `PR`)                              |
| `stat_cause_descr` | Arson, Campfire, Children, Debris Burning, Equipment Use, Fireworks, Lightning, Miscellaneous, Missing/Undefined, Powerline, Railroad, Smoking, Structure |

Invalid input returns a `422` listing the offending fields. `GET /health` returns `{"status": "ok"}`.

## Problem description

Wildfires cause large-scale damage to ecosystems, property and human life, and the resources needed to fight a
fire depend heavily on how big it becomes. Knowing early whether a fire is likely to stay small or grow large helps
with prioritising response and allocating resources.

The project uses the **[1.88 Million US Wildfires](https://www.kaggle.com/datasets/rtatman/188-million-us-wildfires)**
dataset (Short, 2017; US wildfire records from 1992 to 2015, stored as a SQLite database on Kaggle). Fires are
grouped into size classes `A` (smallest) to `G` (largest); this project predicts a **binary target**: is the fire
class `C`-`G` (10+ acres, about 15% of all fires) or not.

| Group    | Features                                                                         |
|----------|----------------------------------------------------------------------------------|
| Time     | year, day of year, month, day of week, discovery hour                            |
| Location | latitude, longitude, state                                                       |
| Cause    | statistical cause (`STAT_CAUSE_DESCR`)                                           |
| Target   | `is_large` (fire size class C-G)                                                 |

The fire size itself is deliberately **not** a feature, since it would leak the answer. Rows with a missing size
class, latitude or longitude are dropped.

## Approach and results

The full analysis (cleaning, EDA, model comparison, tuning) is in [`notebook.ipynb`](notebook.ipynb).

- 60/20/20 stratified train / validation / test split.
- Compared a baseline, logistic regression, decision tree, random forest and XGBoost, tuning each on the
  validation set. Accuracy is misleading with ~15% positives, so models are compared on ROC-AUC and F1 for the
  large class.
- **XGBoost** is the final model: as accurate as the random forest, faster to train and much smaller to deploy.
- The decision threshold is picked on the validation set (best F1) instead of using 0.5.

Final model on the held-out test set (trained on the full 1.88M fires):

| ROC-AUC | Precision | Recall | F1   | Threshold |
|---------|-----------|--------|------|-----------|
| 0.80    | 0.38      | 0.57   | 0.46 | 0.23      |

With only discovery-time information the problem is hard, so the model is best read as a risk ranking: it catches
over half of the large fires, at the cost of many false alarms.

## Project structure

| Path                      | What it is                                                              |
|---------------------------|-------------------------------------------------------------------------|
| `notebook.ipynb`          | Data prep, EDA, model training and selection                            |
| `train.py`                | Trains the final XGBoost model and saves it to `models/wildfire_model.pkl` |
| `predict.py`              | FastAPI service: `/predict`, `/health` and the web UI                   |
| `static/`                 | Web UI (`index.html`) and the US states map data                        |
| `utils/load_data.py`      | Downloads the dataset from Kaggle and caches it as parquet in `data/`   |
| `utils/features.py`       | Feature engineering shared by training and serving                      |
| `models/`                 | The saved model (pipeline, threshold, feature list, test metrics)       |
| `Dockerfile`              | Two-stage build: trains the model if missing, then a small runtime image |
