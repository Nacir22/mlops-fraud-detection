import argparse
import os
import time
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
from scipy.stats import randint


FEATURE_COLS = [
    "amount",
    "balance_delta",
    "type_cash_out",
    "type_payment",
    "type_transfer",
    "type_debit",
    "type_cash_in",
    "step",
]


def load_data(train_path: str, valid_path: str):
    train = pd.read_parquet(train_path)
    valid = pd.read_parquet(valid_path)
    X_train = train[FEATURE_COLS].astype("float32")
    y_train = train["isFraud"].astype(int)
    X_valid  = valid[FEATURE_COLS].astype("float32")
    y_valid  = valid["isFraud"].astype(int)

    return X_train, y_train, X_valid, y_valid


def train_model(X_train, y_train):
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)

    # Espace de recherche (comme demandé)
    param_dist = {
        "max_depth": randint(4, 5),
        "n_estimators": randint(5, 6),
        "class_weight": [None,{0: 1, 1: 2}],
    }

    # CV temporel (3 splits) + AP comme scoring
    tscv = TimeSeriesSplit(n_splits=3)

    start = time.time()
    rand_search = RandomizedSearchCV(
        rf,
        param_distributions=param_dist,
        n_iter=3,                 # comme ta demande
        cv=tscv,
        random_state=42,
        scoring="average_precision",
        n_jobs=-1,
        verbose=1,
    )
    rand_search.fit(X_train, y_train)
    duration = time.time() - start

    best_model = rand_search.best_estimator_
    best_params = rand_search.best_params_
    print("Best hyperparameters:", best_params)
    print(f"Search duration: {duration:.2f}s")

    return best_model, best_params, duration


def main(data_csv: str = None):
    # Si un CSV est donné, on (ré)génère processed + splits
    if data_csv and os.path.exists(data_csv):
        from src.fraud.data_ingest import main as ingest
        from src.fraud.features import main as build_feats
        ingest(data_csv, "data/processed")
        build_feats("data/processed/raw.parquet", "data/processed/features.parquet", 100_000, 0.2)

    X_train, y_train, X_valid, y_valid = load_data(
        "data/processed/train.parquet", "data/processed/valid.parquet"
    )

    mlflow.set_experiment("fraud-detection-rf")
    with mlflow.start_run() as run:
        model, best_params, duration = train_model(X_train, y_train)

        # Évaluation sur le valid set
        proba = model.predict_proba(X_valid)[:, 1]
        y_pred = (proba >= 0.5).astype(int)

        metrics = {
            "auc": float(roc_auc_score(y_valid, proba)),
            "ap": float(average_precision_score(y_valid, proba)),
            "f1": float(f1_score(y_valid, y_pred)),
            "precision": float(precision_score(y_valid, y_pred)),
            "recall": float(recall_score(y_valid, y_pred)),
            "search_seconds": float(duration),
        }

        mlflow.log_params(best_params)
        mlflow.log_metrics(metrics)

        # Enregistrement modèle
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name="fraud_rf",
        )

        print("Run ID:", run.info.run_id)
        print("Metrics:", metrics)
        print("Model registered as 'fraud_rf'")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=False, help="Chemin CSV (optionnel) pour tout rejouer")
    args = ap.parse_args()
    main(args.data)
