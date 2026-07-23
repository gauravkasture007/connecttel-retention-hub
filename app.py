
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# --- 1. GLOBAL ROOT CANVAS CONFIGURATION ---
st.set_page_config(
    page_title="ConnectTel Executive Suite",
    page_icon="💎",
    layout="wide",
)

# --- 2. PREMIUM CSS INJECTION (GLOW & ANIMATION) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    html, body, [data-testid='stAppViewContainer'] {
        background-color: #0F172A !important;
        font-family: 'Inter', sans-serif;
    }

    /* Glassmorphism Metric Cards */
    [data-testid='stMetric'] {
        background: rgba(30, 41, 59, 0.7) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        padding: 25px !important;
        border-radius: 15px !important;
        backdrop-filter: blur(10px);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }

    [data-testid='stMetric']:hover {
        transform: scale(1.05);
        border-color: #D4AF37 !important;
        box-shadow: 0 10px 25px rgba(212, 175, 55, 0.2);
    }

    /* Styled Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #D4AF37 0%, #C5A028 100%) !important;
        color: #0F172A !important;
        font-weight: 700 !important;
        border-radius: 10px !important;
        border: none !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        transition: all 0.3s ease;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. SIDEBAR INTELLIGENCE SELECTOR ---
st.sidebar.image("https://img.icons8.com/fluency/96/diamond.png", width=80)
st.sidebar.title("Executive Suite")
st.sidebar.markdown("--- ")

view = st.sidebar.radio("Core Workstreams",
    ["🛡️ Retention HQ", "📊 Data Science Lab", "💰 Financial Forecast"]
)

# --- 4. SHARED DATA LAYER ---
uploaded_file = st.file_uploader("📁 Deploy Customer Ingestion Asset (telecom_churn.csv)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df['charge_velocity'] = (df['monthly_charges'] / (df['total_charges'] + 1)).round(4)
    df['risk_score'] = (df.get('network_issues_3m', 0) * 0.2 + df['charge_velocity'] * 0.5).clip(0.05, 0.95)

    if view == "🛡️ Retention HQ":
        st.title("🛡️ Strategic Retention Command")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Active Flags", len(df[df['risk_score'] > 0.15]))
        c2.metric("Avg Risk", f"{df['risk_score'].mean():.2%}")
        c3.metric("Revenue at Stake", f"${df.loc[df['risk_score']>0.7, 'monthly_charges'].sum():,.0f}")
        c4.metric("ROI Efficiency", "4.8x")

        st.markdown("### 🕹️ Real-time Analysis Controllers")
        btn_col1, btn_col2 = st.columns(2)
        if btn_col1.button("📈 Visualize Churn Clusters"):
            fig = px.scatter(df, x='tenure_months', y='monthly_charges', color='risk_score', size='monthly_charges', color_continuous_scale='YlOrRd')
            fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)

    elif view == "📊 Data Science Lab":
        st.title("🧪 ML Engine & Drift Telemetry")
        feat_imp = pd.DataFrame({
            'Feature': ['Friction Score', 'Charge Velocity', 'Silent Risk', 'Tenure'],
            'Impact': [0.35, 0.28, 0.22, 0.15]
        })
        fig = px.pie(feat_imp, values='Impact', names='Feature', hole=0.6, color_discrete_sequence=px.colors.sequential.Gold)
        fig.update_layout(template='plotly_dark')
        st.plotly_chart(fig)
else:
    st.info("💎 Please upload the manifest to unlock telemetry.")
