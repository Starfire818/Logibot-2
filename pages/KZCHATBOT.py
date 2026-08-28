# pages/KZCHATBOT.py
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="KZCHATBOX", layout="wide")


if st.button("⬅️ Back to Main Menu"):
    st.switch_page("app.py")

st.subheader("🧑 KZCHATBOX Customer Support")


components.iframe(
    src="https://cdn.botpress.cloud/webchat/v3.6/shareable.html?configUrl=https://files.bpcontent.cloud/2026/07/14/09/20260714093142-YWRPKD7Q.json",
    height=600, 
    scrolling=True
)