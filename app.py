
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import os

st.set_page_config(page_title='ConnectTel Hub', layout='wide')

# Dark Theme CSS
st.markdown("""<style>
    html, body, [data-testid='stAppViewContainer'] { background-color: #0F172A !important; color: #E2E8F0 !important; }
    [data-testid='stMetric'] { background: rgba(30, 41, 59, 0.7); border-radius: 12px; padding: 20px; }
</style>""", unsafe_allow_html=True)

@st.cache_resource
def load_production_artifacts():
    model = joblib.load('connecttel_churn_model.pkl')
    features = joblib.load('feature_columns.pkl')
    return model, features

st.sidebar.title("💎 ConnectTel Suite")
view = st.sidebar.radio("Workstreams", ["🛡️ Retention HQ", "📊 Data Science Lab", "📄 Project Governance"])

if view == "📄 Project Governance":
    st.title("📄 Executive Sign-off & Audit")
    st.markdown("--- ")
    report_file = 'ConnectTel_Executive_Signoff_Report.pdf'
    
    if os.path.exists(report_file):
        with open(report_file, "rb") as f:
            st.download_button(
                label="⏬ Download Executive Sign-off Report (PDF)",
                data=f,
                file_name=report_file,
                mime='application/pdf'
            )
        st.success("✅ Official project documentation is verified and ready for distribution.")
    else:
        st.error("❌ Report artifact missing. Please ensure the PDF is synchronized with the repository.")

elif view == "🛡️ Retention HQ" or view == "📊 Data Science Lab":
    uploaded_file = st.file_uploader("Upload Manifest", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        model, model_features = load_production_artifacts()
        X_input = df.reindex(columns=model_features, fill_value=0)
        df['risk_score'] = model.predict_proba(X_input)[:, 1]

        if view == "🛡️ Retention HQ":
            st.title("🛡️ Retention Command")
            fig = px.scatter(df, x='tenure_months', y='monthly_charges', color='risk_score', color_continuous_scale='Oryel')
            st.plotly_chart(fig, use_container_width=True)

        elif view == "📊 Data Science Lab":
            st.title("🧪 ML Engine Insights")
            feat_data = pd.DataFrame([{'Feature': 'nps_score', 'Power': 0.058867686540538394}, {'Feature': 'avg_data_speed_mbps', 'Power': 0.04840944422709847}, {'Feature': 'charge_velocity', 'Power': 0.04781293943494938}, {'Feature': 'avg_voice_mins_month', 'Power': 0.04708298555330015}, {'Feature': 'total_charges', 'Power': 0.04673949047264355}, {'Feature': 'usage_intensity', 'Power': 0.045471066996841274}, {'Feature': 'dropped_call_rate', 'Power': 0.04503586768433124}, {'Feature': 'service_rating_last_6m', 'Power': 0.04460559144939115}, {'Feature': 'overage_charges', 'Power': 0.04401795119605312}, {'Feature': 'silent_risk_score', 'Power': 0.04369079978160309}])
            fig_bar = px.bar(feat_data, x='Power', y='Feature', orientation='h', color='Power')
            fig_bar.update_layout(template='plotly_dark')
            st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("Please upload data to begin.")
