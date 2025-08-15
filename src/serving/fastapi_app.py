from fastapi import FastAPI
from pydantic import BaseModel
import os
import mlflow
import pandas as pd

app = FastAPI(title="Fraud Detection API")

MODEL_URI = os.getenv("MODEL_URI", "models:/fraud_rf/Production")

class Txn(BaseModel):
    amount: float
    balance_delta: float
    type_cash_out: int
    type_payment: int
    type_transfer: int
    type_debit: int
    type_cash_in: int
    step: int

@app.on_event("startup")
def load_model():
    global model
    model = mlflow.pyfunc.load_model(MODEL_URI)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(txn: Txn):
    X = pd.DataFrame([txn.dict()])
    y = model.predict(X)
    if y.ndim == 1:
        proba = float(y[0])
    else:
        proba = float(y[0,1])
    return {"fraud_probability": proba, "is_fraud": int(proba >= 0.5)}
