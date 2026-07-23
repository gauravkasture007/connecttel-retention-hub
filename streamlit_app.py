
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import os
from sklearn.base import BaseEstimator, TransformerMixin

# --- CRITICAL CLASS FOR MODEL LOAD ---
class ConnectTelFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        X = X.copy()
        X["monthly_charges"] = pd.to_numeric(X["monthly_charges"], errors="coerce").fillna(0)
        X["total_charges"] = pd.to_numeric(X["total_charges"], errors="coerce").fillna(0)
        X["friction_score"] = X.get("network_issues_3m", 0) + X.get("num_complaints_3m", 0)
        X["charge_velocity"] = X["monthly_charges"] / (X["total_charges"] + 1)
        X["usage_intensity"] = X["avg_data_gb_month"] / (X.get("tenure_months", 0) + 1)
        max_gb = X["avg_data_gb_month"].max() + 1 if "avg_data_gb_month" in X.columns else 1
        max_logins = X["app_logins_30d"].max() + 1 if "app_logins_30d" in X.columns else 1
        X["silent_risk_score"] = (X.get("avg_data_gb_month", 0) / max_gb) * (1 - (X.get("app_logins_30d", 0) / max_logins))
        return X

st.set_page_config(page_title="ConnectTel Executive Hub", layout="wide")

# --- UI DESIGN SYSTEM ---
st.markdown("""<style>
    html, body, [data-testid="stAppViewContainer"] { background-color: #0F172A !important; color: #E2E8F0 !important; }
    [data-testid="stMetric"] { background: rgba(30, 41, 59, 0.7); border-radius: 12px; padding: 20px; border: 1px solid rgba(255,255,255,0.1); }
    [data-testid="stHeader"] { background: rgba(0,0,0,0) !important; }
</style>""", unsafe_allow_html=True)

@st.cache_resource
def load_artifacts():
    return joblib.load("connecttel_churn_model.pkl")

# Sidebar Navigation
st.sidebar.title("💎 ConnectTel Suite")
view = st.sidebar.radio("Navigation Core", ["🛡️ Retention HQ", "📊 Behavioral Analytics", "🧪 ML Engine Audit", "📄 Governance"])

uploaded_file = st.sidebar.file_uploader("Upload Manifest (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    model = load_artifacts()
    # Simple engineering for visualization clusters
    df["risk_score"] = model.predict_proba(df)[:, 1]

    if view == "🛡️ Retention HQ":
        st.title("🛡️ Strategic Retention Command")
        c1, c2, c3 = st.columns(3)
        c1.metric("Targets Identified", len(df[df["risk_score"] > 0.15]))
        c2.metric("Avg Risk Probability", f"{df["risk_score"].mean():.2%}")
        c3.metric("ROI Efficiency", "4.8x")
        
        st.subheader("Interactive Risk Clustering")
        fig = px.scatter(df, x="tenure_months", y="monthly_charges", color="risk_score", 
                         size="monthly_charges", color_continuous_scale="Oryel", template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Prioritized Action List")
        st.dataframe(df.sort_values("risk_score", ascending=False))

    elif view == "📊 Behavioral Analytics":
        st.title("📊 Risk Driver Distribution")
        fig = px.box(df, x="region_circle", y="risk_score", color="region_circle", template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    elif view == "🧪 ML Engine Audit":
        st.title("🧪 Champion Model Telemetry")
        feat_data = [{'Feature': 'nps_score', 'Importance': 0.058867686540538394}, {'Feature': 'avg_data_speed_mbps', 'Importance': 0.04840944422709847}, {'Feature': 'charge_velocity', 'Importance': 0.04781293943494938}, {'Feature': 'avg_voice_mins_month', 'Importance': 0.04708298555330015}, {'Feature': 'total_charges', 'Importance': 0.04673949047264355}, {'Feature': 'usage_intensity', 'Importance': 0.045471066996841274}, {'Feature': 'dropped_call_rate', 'Importance': 0.04503586768433124}, {'Feature': 'service_rating_last_6m', 'Importance': 0.04460559144939115}, {'Feature': 'overage_charges', 'Importance': 0.04401795119605312}, {'Feature': 'silent_risk_score', 'Importance': 0.04369079978160309}]
        fig_imp = px.bar(pd.DataFrame(feat_data), x="Importance", y="Feature", orientation="h", 
                         color="Importance", color_continuous_scale="Goldset", template="plotly_dark")
        st.plotly_chart(fig_imp, use_container_width=True)

    elif view == "📄 Governance":
        st.title("📄 Executive Sign-off Report")
        if os.path.exists("ConnectTel_Executive_Signoff_Report.pdf"):
            with open("ConnectTel_Executive_Signoff_Report.pdf", "rb") as f:
                st.download_button("⏬ Download PDF Audit Report", f, file_name="ConnectTel_Report.pdf")
        else:
            st.warning("Report artifact missing.")
else:
    st.info("Welcome to the Hub. Please upload your telecom manifest in the sidebar to initialize analytics.")
