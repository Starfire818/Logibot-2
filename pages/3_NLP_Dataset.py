"""
File: pages/3_NLP_Dataset.py
Description: Backward-compatible redirect page. The former combined NLP Dataset + Model
             Evaluation page was split into two separate pages (assignment §18 suggestion
             of 5 distinct UI pages instead of one large page) plus a dedicated About page:
               - pages/3_Model_Evaluation.py
               - pages/4_Dataset_Analysis.py
               - pages/5_About_LogiBot.py
             This redirect page keeps any existing bookmarks / sidebar links working.
"""

import streamlit as st

st.set_page_config(
    page_title="LogiBot - Dataset & Model (redirect)",
    page_icon="🔀",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🔀 Page reorganized")
st.info(
    "To match the assignment §18 recommended 5-page UI structure, the previous combined "
    "Dataset + Model Evaluation page was split into three dedicated pages. Click below to "
    "navigate to the new pages:"
)

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("### 📊 Model Evaluation")
    st.caption("Accuracy / P / R / F1, classification report, confusion matrix, model comparison (LinearSVC vs LogReg), holdout predictions.")
    if st.button("Open Model Evaluation →", type="primary", use_container_width=True):
        st.switch_page("pages/3_Model_Evaluation.py")
with c2:
    st.markdown("### 📁 Dataset Analysis")
    st.caption("Total/train/test samples, intent names, samples per intent chart, train/test split, data quality, robustness probes, user feedback.")
    if st.button("Open Dataset Analysis →", type="primary", use_container_width=True):
        st.switch_page("pages/4_Dataset_Analysis.py")
with c3:
    st.markdown("### 📘 About LogiBot (report-ready docs)")
    st.caption("§26 6-chapter narrative: Intro / Related Work / Methodology / Results & Discussion / Conclusion / References, plus live user satisfaction summary.")
    if st.button("Open About LogiBot →", type="primary", use_container_width=True):
        st.switch_page("pages/5_About_LogiBot.py")

st.markdown("---")
st.sidebar.markdown("### Quick navigation")
if st.sidebar.button("← Home / Start chat"):
    st.switch_page("app.py")
if st.sidebar.button("💬 Chat"):
    st.switch_page("pages/1_Chat.py")
if st.sidebar.button("🔍 NLP Analysis"):
    st.switch_page("pages/2_NLP_Analysis.py")
