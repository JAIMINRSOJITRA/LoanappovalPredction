import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# =========================
# Page Config
# =========================
st.set_page_config(
    page_title="Loan Approval Prediction System",
    layout="wide",
    page_icon="🏦",
)

# =========================
# Custom CSS (Modern Light + Cards + Glassmorphism)
# =========================
st.markdown(
    """
    <style>
    body {
        background-color: #f5f7fb;
    }
    .main {
        background-color: #f5f7fb;
    }
    .big-title {
        font-size: 2.4rem;
        font-weight: 700;
        color: #1f3b64;
    }
    .subtitle {
        font-size: 1rem;
        color: #4a4a4a;
    }
    .card {
        background: #ffffff;
        border-radius: 16px;
        padding: 20px 24px;
        box-shadow: 0 8px 16px rgba(15, 23, 42, 0.08);
        margin-bottom: 16px;
    }
    .metric-card {
        background: #ffffff;
        border-radius: 16px;
        padding: 16px 18px;
        box-shadow: 0 4px 10px rgba(15, 23, 42, 0.06);
    }
    .hero-card {
        border-radius: 18px;
        padding: 24px 28px;
        background: linear-gradient(135deg, #e0f2fe, #f5f3ff);
        box-shadow: 0 12px 24px rgba(37, 99, 235, 0.25);
    }
    .glass-card {
        border-radius: 18px;
        padding: 24px 28px;
        background: rgba(255, 255, 255, 0.85);
        box-shadow: 0 8px 16px rgba(15, 23, 42, 0.12);
        backdrop-filter: blur(8px);
    }
    .footer {
        text-align: center;
        color: #6b7280;
        font-size: 0.9rem;
        padding: 18px 0px 4px 0px;
    }
    .stButton>button {
        border-radius: 999px;
        padding: 0.6rem 1.6rem;
        font-weight: 600;
        border: none;
        background: linear-gradient(135deg, #2563eb, #4f46e5);
        color: white;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #1d4ed8, #4338ca);
    }
    .sidebar-title {
        font-size: 1.2rem !important;
        font-weight: 700 !important;
        color: #1f2937 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================
# Helper Functions
# =========================
@st.cache_resource
def load_pipeline(model_path: str = "loan_pipeline.joblib"):
    """Load the trained sklearn pipeline."""
    if not os.path.exists(model_path):
        return None, f"⚠️ Model file `{model_path}` not found!"
    try:
        pipe = joblib.load(model_path)
        return pipe, None
    except Exception as e:
        return None, f"❌ Failed to load model: {e}"


@st.cache_data
def load_sample_data(csv_path: str = "loan_approval_dataset.csv"):
    """Load sample dataset for insights page if available."""
    if not os.path.exists(csv_path):
        return None, f"Dataset `{csv_path}` not found."
    try:
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip().str.lower()
        return df, None
    except Exception as e:
        return None, f"Failed to load dataset: {e}"


def build_cibil_gauge(cibil_score: int):
    score = max(300, min(900, cibil_score))
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            title={"text": "CIBIL Score", "font": {"size": 18}},
            gauge={
                "axis": {"range": [300, 900]},
                "bar": {"color": "#2563eb"},
                "steps": [
                    {"range": [300, 600], "color": "#fee2e2"},
                    {"range": [600, 750], "color": "#fef3c7"},
                    {"range": [750, 900], "color": "#dcfce7"},
                ],
                "threshold": {
                    "line": {"color": "#16a34a", "width": 4},
                    "thickness": 0.75,
                    "value": score,
                },
            },
        )
    )
    fig.update_layout(height=260)
    return fig


def applicant_summary_card(data: dict, prediction_label: str):
    st.markdown(f"""
    **👤 Applicant Summary**
    - Dependents: **{data['no_of_dependents']}**
    - Education: **{"Graduate" if data['education'] == 1 else "Not Graduate"}**
    - Self Employed: **{"Yes" if data['self_employed'] == 1 else "No"}**
    - Annual Income: **₹{data['income_annum']:,.0f}**
    - Loan Amount: **₹{data['loan_amount']:,.0f}**
    - Loan Term: **{data['loan_term']} years**
    - CIBIL Score: **{data['cibil_score']}**
    - Status (Model): **{prediction_label}**
    """)


# =========================
# Sidebar Navigation
# =========================
st.sidebar.markdown("<p class='sidebar-title'>🏦 Loan Dashboard</p>", unsafe_allow_html=True)
page = st.sidebar.radio("Navigate", ["Home", "Predict Loan", "Insights", "About"])
st.sidebar.info("Use this app to estimate if a loan is likely to be **approved**.")

pipeline, model_error = load_pipeline()

# =========================
# HOME PAGE
# =========================
if page == "Home":
    st.markdown("<div class='hero-card'>", unsafe_allow_html=True)
    st.markdown("<div class='big-title'>🏦 Loan Approval Prediction System</div>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Analyze patterns & predict loan approvals in seconds.</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='footer'>Made with ❤️ by <b>Jaimin Sojitra</b></div>", unsafe_allow_html=True)

# =========================
# PREDICT LOAN PAGE
# =========================
elif page == "Predict Loan":
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("📥 Enter Applicant Details")

    left_col, right_col = st.columns(2)

    with left_col:
        no_of_dependents = st.number_input("Dependents", 0, step=1)
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        self_employed = st.selectbox("Self Employed", ["Yes", "No"])
        cibil_score = st.number_input("CIBIL Score", 300, 900, 750)

    with right_col:
        income_annum = st.number_input("Annual Income (₹)", 0, step=10000)
        loan_amount = st.number_input("Loan Amount (₹)", 0, step=10000)
        loan_term = st.number_input("Loan Term", 1, 40, 10)
        residential_assets = st.number_input("Residential Assets (₹)", 0)
        commercial_assets = st.number_input("Commercial Assets (₹)", 0)
        luxury_assets = st.number_input("Luxury Assets (₹)", 0)
        bank_assets = st.number_input("Bank Assets (₹)", 0)

    edu_encoded = 1 if education == "Graduate" else 0
    se_encoded = 1 if self_employed == "Yes" else 0

    input_data = {
        "no_of_dependents": no_of_dependents,
        "education": edu_encoded,
        "self_employed": se_encoded,
        "income_annum": float(income_annum),
        "loan_amount": float(loan_amount),
        "loan_term": float(loan_term),
        "cibil_score": int(cibil_score),
        "residential_assets_value": float(residential_assets),
        "commercial_assets_value": float(commercial_assets),
        "luxury_assets_value": float(luxury_assets),
        "bank_asset_value": float(bank_assets),
    }

    input_df = pd.DataFrame([input_data])

    if st.button("🔮 Predict Loan Approval"):
        if model_error:
            st.error(model_error)
        else:
            try:
                pred = pipeline.predict(input_df)[0]
                proba = pipeline.predict_proba(input_df)[0][1]

                prediction_label = "Approved" if pred == 1 else "Rejected"

                if pred == 1:
                    st.success(f"🎉 Loan Approved! — Probability: {proba*100:.2f}%")
                else:
                    st.error(f"❌ Loan Rejected! — Probability: {(1-proba)*100:.2f}%")

                st.write("---")
                applicant_summary_card(input_data, prediction_label)

            except Exception as e:
                st.error(f"⚠️ Prediction failed: {e}")

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<div class='footer'>Made with ❤️ by <b>Jaimin Sojitra</b></div>", unsafe_allow_html=True)

# =========================
# INSIGHTS PAGE
# =========================
elif page == "Insights":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📈 Loan Dataset Insights")

    df, data_error = load_sample_data()
    if data_error:
        st.error(data_error)
    else:
        st.write(df.head())

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<div class='footer'>Made with ❤️ by <b>Jaimin Sojitra</b></div>", unsafe_allow_html=True)

# =========================
# ABOUT PAGE
# =========================
elif page == "About":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("👤 About this App")

    st.write("""
    This ML model predicts loan approval using details like income, CIBIL score, and assets.
    Designed with passion ❤️ and powered by Machine Learning.
    """)

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<div class='footer'>Made with ❤️ by <b>Jaimin Sojitra</b></div>", unsafe_allow_html=True)
