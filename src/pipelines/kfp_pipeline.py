import argparse
import kfp
from kfp import dsl

@dsl.component(base_image="python:3.10-slim")
def ingest_op(data_path: str) -> str:
    import subprocess, sys, os
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas", "pyarrow"])
    from pathlib import Path
    import pandas as pd
    out_dir = "data/processed"
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(data_path)
    keep = [
        "step","type","amount","nameOrig","oldbalanceOrg","newbalanceOrig",
        "nameDest","oldbalanceDest","newbalanceDest","isFraud"
    ]
    df = df[keep]
    df.to_parquet(f"{out_dir}/raw.parquet", index=False)
    return f"{out_dir}/raw.parquet"

@dsl.component(base_image="python:3.10-slim")
def features_op(in_parquet: str) -> str:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas", "pyarrow"])
    import pandas as pd
    from pathlib import Path
    df = pd.read_parquet(in_parquet)
    df["balance_delta"] = (df["newbalanceOrig"] - df["oldbalanceOrg"]).fillna(0.0)
    types = ["CASH_OUT","PAYMENT","TRANSFER","DEBIT","CASH_IN"]
    for t in types:
        df[f"type_{t.lower()}"] = (df["type"] == t).astype("int64")
    df["event_timestamp"] = pd.to_datetime(df["step"], unit="h", origin="unix")
    feat_cols = [
        "amount","balance_delta","type_cash_out","type_payment",
        "type_transfer","type_debit","type_cash_in","step","isFraud"
    ]
    feats = df[feat_cols]
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    feats.to_parquet("data/processed/features.parquet", index=False)
    return "data/processed/features.parquet"

@dsl.component(base_image="python:3.10-slim")
def train_op(features_path: str) -> str:
    import subprocess, sys, os
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas", "scikit-learn", "xgboost", "mlflow", "pyarrow"])
    import pandas as pd
    import xgboost as xgb
    from sklearn.metrics import roc_auc_score
    import mlflow
    df = pd.read_parquet(features_path)
    X = df.drop(columns=["isFraud"])
    y = df["isFraud"].astype(int)
    split = int(0.8 * len(df))
    X_train, y_train = X.iloc[:split], y.iloc[:split]
    X_valid, y_valid = X.iloc[split:], y.iloc[split:]
    n_pos = max(int((y_train == 1).sum()), 1)
    n_neg = max(int((y_train == 0).sum()), 1)
    scale_pos_weight = n_neg / n_pos
    params = dict(n_estimators=200, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, objective="binary:logistic", eval_metric="auc", scale_pos_weight=scale_pos_weight)
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI","http://mlflow:5000"))
    mlflow.set_experiment("fraud-detection")
    with mlflow.start_run() as run:
        model = xgb.XGBClassifier(**params).fit(X_train, y_train)
        auc = roc_auc_score(y_valid, model.predict_proba(X_valid)[:,1])
        mlflow.log_metric("auc", float(auc))
        mlflow.sklearn.log_model(model, "model", registered_model_name="fraud_xgb")
        print("AUC:", auc)
        return run.info.run_id

@dsl.component(base_image="python:3.10-slim")
def drift_op(features_path: str):
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas", "evidently", "pyarrow"])
    import pandas as pd
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset
    df = pd.read_parquet(features_path)
    split = int(0.8 * len(df))
    ref = df.iloc[:split].drop(columns=["isFraud"])
    cur = df.iloc[split:].drop(columns=["isFraud"])
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref, current_data=cur)
    report.save_html("drift_report.html")
    print("Drift report saved")

@dsl.pipeline(name="fraud-detection-pipeline", description="Ingest -> Features -> Train -> Drift")
def pipeline(csv_path: str = "data/raw/PS_20174392719_1491204439457_log.csv"):
    raw = ingest_op(csv_path)
    feats = features_op(raw.outputs["output"])
    train = train_op(feats.outputs["output"])
    drift_op(feats.outputs["output"])

def compile_pipeline(outfile: str):
    kfp.compiler.Compiler().compile(pipeline, outfile)

def main(compile_path: str = None, submit: bool = False):
    if compile_path:
        compile_pipeline(compile_path)
    if submit:
        client = kfp.Client()  # assumes KFP endpoint from env
        client.create_run_from_pipeline_func(pipeline, arguments={})

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--compile", dest="compile_path", default=None, help="Path to output pipeline.yaml")
    ap.add_argument("--submit", action="store_true")
    args = ap.parse_args()
    main(args.compile_path, args.submit)
