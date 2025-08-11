import argparse
import os
import mlflow
import mlflow.sklearn
import xgboost as xgb
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from src.fraud.utils import compute_scale_pos_weight

def load_data(train_path: str, valid_path: str):
    train = pd.read_parquet(train_path)
    valid = pd.read_parquet(valid_path)
    feature_cols = [
        "amount","balance_delta","type_cash_out","type_payment",
        "type_transfer","type_debit","type_cash_in","step"
    ]
    X_train = train[feature_cols]
    y_train = train["isFraud"].astype(int)
    X_valid = valid[feature_cols]
    y_valid = valid["isFraud"].astype(int)
    return X_train, y_train, X_valid, y_valid

def main(data_csv: str):
    # If user points to CSV, generate processed data
    if data_csv and os.path.exists(data_csv):
        from src.fraud.data_ingest import main as ingest
        from src.fraud.features import main as build_feats
        ingest(data_csv, "data/processed")
        build_feats("data/processed/raw.parquet", "data/processed/features.parquet")

    X_train, y_train, X_valid, y_valid = load_data("data/processed/train.parquet","data/processed/valid.parquet")

    scale_pos_weight = compute_scale_pos_weight(y_train)
    params = dict(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="auc",
        scale_pos_weight=scale_pos_weight,
        n_jobs=-1,
        reg_lambda=1.0,
    )

    mlflow.set_experiment("fraud-detection")
    with mlflow.start_run() as run:
        mlflow.log_params(params)
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)

        preds = model.predict_proba(X_valid)[:,1]
        preds_label = (preds >= 0.5).astype(int)

        metrics = {
            "auc": float(roc_auc_score(y_valid, preds)),
            "f1": float(f1_score(y_valid, preds_label)),
            "precision": float(precision_score(y_valid, preds_label)),
            "recall": float(recall_score(y_valid, preds_label)),
        }
        mlflow.log_metrics(metrics)

        # Log model
        mlflow.sklearn.log_model(model, artifact_path="model", registered_model_name="fraud_xgb")

        print("Run ID:", run.info.run_id)
        print("Metrics:", metrics)
        print("Model registered as 'fraud_xgb'")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=False, help="CSV dataset path; if provided, pipeline runs end-to-end.")
    args = ap.parse_args()
    main(args.data)
