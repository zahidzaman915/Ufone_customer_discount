import os, sys
import joblib
import pandas as pd
import streamlit as st

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE, "src"))
from feature_engineering import add_features  # required so joblib can load the pipeline

@st.cache_resource
def load_model():
    return joblib.load(os.path.join(BASE, "model", "model_pipeline.pkl"))

bundle = load_model()
pipe, threshold = bundle["pipeline"], bundle["threshold"]

st.set_page_config(page_title="Ufone Discount Targeting", layout="wide")
st.title("Ufone Customer Discount Targeting System")
st.write("Estimates whether a customer's historical behaviour resembles customers who are "
         "good candidates for a discounted internet offer.")

with st.form("customer_form"):
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.number_input("Age", 18, 70, 32)
        region = st.selectbox("Region", ["Lahore", "Karachi", "Islamabad", "Rawalpindi",
                                         "Faisalabad", "Multan", "Peshawar", "Gujranwala"])
        tenure = st.number_input("Tenure (months)", 1, 120, 24)
        device = st.selectbox("Device type", ["4G Smartphone", "Basic/2G Phone"])
        network = st.slider("Network quality score (3-5)", 3.0, 5.0, 4.0, 0.1)
        voice = st.number_input("Avg monthly voice minutes", 0.0, 3000.0, 150.0)
        sms = st.number_input("Avg monthly SMS", 0, 1000, 15)
    with c2:
        data_gb = st.number_input("Avg monthly data GB (before last 60 days)", 0.0, 200.0, 2.5)
        acts = st.number_input("Package activations (before last 60 days)", 0, 50, 2)
        tier = st.selectbox("Typical package price tier", ["Low", "Medium", "High"], index=1)
        trend = st.number_input("Data usage trend (%)", -100.0, 200.0, 0.0)
        recharge = st.number_input("Avg monthly recharge (PKR)", 0.0, 20000.0, 400.0)
        freq = st.number_input("Recharges per month", 0, 12, 3)
    with c3:
        days = st.number_input("Days since last recharge", 0, 120, 10)
        variability = st.number_input("Recharge amount variability (0-1)", 0.0, 1.0, 0.25)
        spend_trend = st.number_input("Monthly spend trend (%)", -100.0, 200.0, 0.0)
        app_use = st.number_input("App sessions per month", 0, 500, 8)
        complaints = st.number_input("Complaints count", 0, 20, 0)
    submitted = st.form_submit_button("Predict")

if submitted:
    row = pd.DataFrame([{
        "age": age, "region": region, "customer_tenure_months": tenure,
        "device_type": device, "network_quality_score": network,
        "avg_monthly_voice_minutes": voice, "avg_monthly_sms": sms,
        "avg_monthly_data_gb": data_gb, "pre_window_package_activations": acts,
        "avg_package_price_tier": tier, "data_usage_trend": trend,
        "avg_monthly_recharge_amount": recharge, "recharge_frequency": freq,
        "days_since_last_recharge": days, "recharge_amount_variability": variability,
        "monthly_spend_trend": spend_trend, "app_usage_frequency": app_use,
        "complaints_count": complaints}])
    score = pipe.predict_proba(row)[0, 1]
    pred = int(score >= threshold)
    st.subheader(f"Prediction: {pred}")
    if pred == 1:
        st.success("Discount Offer Recommended")
        st.write("This customer's behaviour resembles customers who are potential candidates "
                 "for a discount offer.")
    else:
        st.info("No Discount Offer Required")
        st.write("This customer's behaviour resembles an active internet customer, "
                 "so a discount may not be necessary.")
    st.metric("Candidate score", f"{score:.0%}")
    st.caption(f"The score is a model estimate (class-balanced), not a guaranteed probability. "
               f"Offer is recommended when the score is at least {threshold:.0%}.")

st.divider()
st.caption("This project uses synthetic data for educational/bootcamp purposes. Predictions are "
           "model estimates and do not guarantee customer behavior. This is not an official Ufone system.")
