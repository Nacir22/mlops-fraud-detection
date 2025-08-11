import argparse
import mlflow
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

def main(model_uri: str, data_path: str = "data/processed/valid.parquet"):
    model = mlflow.pyfunc.load_model(model_uri)
    df = pd.read_parquet(data_path)
    feature_cols = [
        "amount","balance_delta","type_cash_out","type_payment",
        "type_transfer","type_debit","type_cash_in","step"
    ]
    X = df[feature_cols]
    y = df["isFraud"].astype(int).values
    proba = model.predict(X)
    # mlflow pyfunc may return proba or labels depending on flavor; ensure proba
    if proba.ndim == 1:
        preds = (proba >= 0.5).astype(int)
    else:
        preds = (proba[:,1] >= 0.5).astype(int)
    print(classification_report(y, preds))
    print("Confusion matrix:\n", confusion_matrix(y, preds))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_uri", required=True, help="e.g., models:/fraud_xgb/Production or runs:/<runid>/model")
    ap.add_argument("--data", default="data/processed/valid.parquet")
    args = ap.parse_args()
    main(args.model_uri, args.data)
