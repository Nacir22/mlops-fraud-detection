import streamlit as st
import requests
import os

st.set_page_config(page_title="Fraud Detection", page_icon="🕵️")

st.title("🕵️ Fraud Detection Demo")

api_url = st.text_input("FastAPI endpoint", os.getenv("API_URL","http://localhost:8000/predict"))

col1, col2 = st.columns(2)
with col1:
    amount = st.number_input("amount", min_value=0.0, value=100.0)
    balance_delta = st.number_input("balance_delta", value=-50.0)
    step = st.number_input("step (hours)", min_value=0, value=1)
with col2:
    type_cash_out = st.selectbox("type_cash_out", [0,1], index=0)
    type_payment = st.selectbox("type_payment", [0,1], index=0)
    type_transfer = st.selectbox("type_transfer", [0,1], index=0)
    type_debit = st.selectbox("type_debit", [0,1], index=0)
    type_cash_in = st.selectbox("type_cash_in", [0,1], index=0)

if st.button("Predict"):
    payload = {
        "amount": amount,
        "balance_delta": balance_delta,
        "type_cash_out": int(type_cash_out),
        "type_payment": int(type_payment),
        "type_transfer": int(type_transfer),
        "type_debit": int(type_debit),
        "type_cash_in": int(type_cash_in),
        "step": int(step),
    }
    try:
        r = requests.post(api_url, json=payload, timeout=10)
        st.write(r.json())
    except Exception as e:
        st.error(str(e))
