"""Train the final wildfire-severity model (XGBoost) and save it to disk.

Usage:
    python train.py              # train on the full dataset (~1.88M fires)
    python train.py --sample     # quick run on the 500k sample

Same recipe as the notebook: 60/20/20 stratified split, XGBoost with the tuned
hyperparameters, decision threshold picked on validation (best F1), metrics
reported once on the untouched test set.
"""
import argparse
import json
import pickle
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBClassifier

from utils.features import CATEGORICAL, FEATURES, prepare_training_data
from utils.load_data import load_data

RANDOM_STATE = 42
MODEL_PATH = Path(__file__).resolve().parent / "models" / "wildfire_model.pkl"

XGB_PARAMS = dict(
    n_estimators=300, learning_rate=0.1, max_depth=8, min_child_weight=10,
    subsample=0.8, colsample_bytree=0.8, tree_method="hist", n_jobs=-1,
    random_state=RANDOM_STATE,
)


def best_f1_threshold(y_true, proba) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    return float(thresholds[f1[:-1].argmax()])


def build_pipeline():
    # Categories -> integer codes (unseen -> -1); numbers pass through (XGBoost handles NaN)
    preprocess = ColumnTransformer(
        [("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CATEGORICAL)],
        remainder="passthrough",
    )
    return make_pipeline(preprocess, XGBClassifier(**XGB_PARAMS))


def main(full: bool, output: Path):
    X, y = prepare_training_data(load_data(full=full))

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=RANDOM_STATE)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=RANDOM_STATE)
    print(f"train: {len(X_train):,}  val: {len(X_val):,}  test: {len(X_test):,}")

    model = build_pipeline().fit(X_train, y_train)

    threshold = best_f1_threshold(y_val, model.predict_proba(X_val)[:, 1])

    test_proba = model.predict_proba(X_test)[:, 1]
    test_pred = test_proba >= threshold
    metrics = {
        "roc_auc": roc_auc_score(y_test, test_proba),
        "threshold": threshold,
        "precision": precision_score(y_test, test_pred),
        "recall": recall_score(y_test, test_pred),
        "f1": f1_score(y_test, test_pred),
    }
    print("Test metrics:", json.dumps({k: round(v, 3) for k, v in metrics.items()}, indent=2))

    # Everything the serving code needs lives in one file
    artifact = {"model": model, "threshold": threshold, "features": FEATURES, "metrics": metrics}
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "wb") as f:
        pickle.dump(artifact, f)
    print(f"Saved model to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", action="store_true", help="train on the 500k sample instead of the full data")
    parser.add_argument("--output", type=Path, default=MODEL_PATH)
    args = parser.parse_args()
    main(full=not args.sample, output=args.output)
