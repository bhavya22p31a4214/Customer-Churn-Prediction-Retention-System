import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model
model = joblib.load("model/churn_model.pkl")

# Page config
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide"
)

# ---------- CUSTOM CSS ----------
st.markdown("""
<style>
body {
    background-color: #f5f9ff;
}
.main {
    background-color: #f5f9ff;
}
.header {
    background: linear-gradient(90deg, #0a58ca, #1e90ff);
    padding: 20px;
    border-radius: 12px;
    color: white;
    text-align: center;
}
.card {
    background-color: white;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}
.result-high {
    background-color: #ffe5e5;
    padding: 15px;
    border-radius: 10px;
    border-left: 6px solid #dc3545;
}
.result-low {
    background-color: #e7f5ff;
    padding: 15px;
    border-radius: 10px;
    border-left: 6px solid #0d6efd;
}
.footer {
    text-align: center;
    color: gray;
    font-size: 13px;
}
</style>
""", unsafe_allow_html=True)

# ---------- HEADER ----------
st.markdown("""
<div class="header">
    <h1>📊 Customer Churn Prediction System</h1>
    <p>AI-powered decision support tool for customer retention</p>
</div>
""", unsafe_allow_html=True)

st.write("")
st.write("")

# ---------- LAYOUT ----------
left, right = st.columns(2)

# ---------- INPUT CARD ----------
with left:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🧾 Customer Information")

    tenure = st.number_input("Tenure (months)", 0, 100, 12)
    monthly_charges = st.number_input("Monthly Charges", 0.0, 500.0, 70.0)
    total_charges = st.number_input("Total Charges", 0.0, 10000.0, 1000.0)

    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    payment_method = st.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
    )

    predict_btn = st.button("🔍 Predict Churn", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ---------- ENCODING ----------
contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
internet_map = {"DSL": 0, "Fiber optic": 1, "No": 2}
payment_map = {
    "Electronic check": 0,
    "Mailed check": 1,
    "Bank transfer (automatic)": 2,
    "Credit card (automatic)": 3
}

input_df = pd.DataFrame([[
    tenure,
    monthly_charges,
    total_charges,
    contract_map[contract],
    internet_map[internet_service],
    payment_map[payment_method]
]])

# ---------- RESULT CARD ----------
with right:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📈 Prediction Result")

    if predict_btn:
        n_features = model.n_features_in_
        if input_df.shape[1] < n_features:
            input_data = np.hstack([
                input_df.values,
                np.zeros((1, n_features - input_df.shape[1]))
            ])
        else:
            input_data = input_df.values

        prob = model.predict_proba(input_data)[0][1]
        threshold = 0.7

        st.metric("Churn Probability", f"{prob*100:.2f}%")

        if prob >= threshold:
            st.markdown("""
            <div class="result-high">
                <b>⚠️ High Risk:</b> Customer is likely to churn<br><br>
                <b>Recommendation:</b> Offer discounts, improve support, or suggest long-term plans.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="result-low">
                <b>✅ Low Risk:</b> Customer likely to stay<br><br>
                <b>Recommendation:</b> Maintain service quality and engagement.
            </div>
            """, unsafe_allow_html=True)

        st.info(
            "This is a demo interface with limited features. "
            "In a production environment, all customer attributes would be captured."
        )

    st.markdown("</div>", unsafe_allow_html=True)

# ---------- FOOTER ----------
st.markdown("""
<div class="footer">
    Built with Python • Machine Learning • Streamlit
</div>
""", unsafe_allow_html=True)
