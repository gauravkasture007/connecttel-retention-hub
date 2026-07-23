import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="ConnectTel Executive Hub", layout="wide")

st.sidebar.title("ሳ ConnectTel Suite")
view_selection = st.sidebar.radio("Navigation", 
    ["ሓ Retention Center", "ሲ Executive Sign-off Report"])

if view_selection == "ሲ Executive Sign-off Report":
    st.title("ሲ Executive Strategy & Sign-off")
    report_path = 'ConnectTel_Executive_Signoff_Report.pdf'
    if os.path.exists(report_path):
        with open(report_path, "rb") as pdf_file:
            st.download_button("ሲ Download PDF Report", pdf_file, file_name=report_path)
        st.success("ሴ Final Executive Report is available for download.")
    else:
        st.error("ስ Report artifact not found in repository.")
else:
    st.title("ሓ Customer Retention Intelligence")
    st.info("Please upload your telecom data to begin analysis.")
