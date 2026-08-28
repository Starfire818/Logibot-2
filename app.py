"""
File: app.py
Description: Main landing page for Smart Logistics Assistant.
             Displays available chatbot options with a modern card-based UI.
"""

import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Smart Logistics Assistant",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
* {
  box-sizing: border-box;
}

:root {
  color-scheme: light dark;
}

/* Light Mode */
body, .stApp, .main, .block-container {
  background: linear-gradient(135deg, #F0F9FF 0%, #FFFFFF 100%) !important;
}

#MainMenu, footer {
  visibility: hidden;
}

/* Main container */
.main-container {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  min-height: 100vh;
  padding: 40px 20px;
}

/* Header */
.header-section {
  text-align: center;
  margin-bottom: 60px;
}

.header-title {
  font-size: 3rem;
  font-weight: 800;
  background: linear-gradient(135deg, #2563EB 0%, #60A5FA 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  margin: 0 0 16px 0;
  line-height: 1.1;
}

.header-subtitle {
  font-size: 1.2rem;
  color: #64748B;
  margin: 0;
  font-weight: 500;
}

/* Cards Container */
.cards-container {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 24px;
  max-width: 900px;
  width: 100%;
  margin-top: 40px;
}

/* Card Styling */
.chatbot-card {
  background: white;
  border-radius: 20px;
  padding: 32px;
  box-shadow: 0 10px 40px rgba(15, 23, 42, 0.1);
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  border: 2px solid transparent;
  cursor: pointer;
}

.chatbot-card:hover {
  transform: translateY(-8px);
  box-shadow: 0 20px 60px rgba(37, 99, 235, 0.2);
  border-color: #2563EB;
}

.chatbot-card.disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.chatbot-card.disabled:hover {
  transform: none;
  box-shadow: 0 10px 40px rgba(15, 23, 42, 0.1);
  border-color: transparent;
}

.card-icon {
  font-size: 3rem;
  margin-bottom: 16px;
}

.card-title {
  font-size: 1.5rem;
  font-weight: 700;
  color: #0F172A;
  margin: 0 0 8px 0;
}

.card-description {
  font-size: 0.95rem;
  color: #64748B;
  margin: 0 0 24px 0;
  line-height: 1.5;
}

.card-button {
  width: 100%;
  padding: 12px 24px;
  border: none;
  border-radius: 12px;
  font-size: 1rem;
  font-weight: 600;
  transition: all 0.2s ease;
  cursor: pointer;
}

.card-button.enabled {
  background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%);
  color: white;
}

.card-button.enabled:hover {
  transform: translateY(-2px);
  box-shadow: 0 8px 24px rgba(37, 99, 235, 0.3);
}

.card-button.enabled:active {
  transform: translateY(0);
}

.card-button.disabled {
  background: #E2E8F0;
  color: #94A3B8;
  cursor: not-allowed;
}

/* Footer */
.footer-text {
  text-align: center;
  color: #94A3B8;
  font-size: 0.9rem;
  margin-top: 60px;
}

/* Dark Mode */
@media (prefers-color-scheme: dark) {
  body, .stApp, .main, .block-container {
    background: linear-gradient(135deg, #0B1220 0%, #0F172A 100%) !important;
    color: #F8FAFC !important;
  }
  
  .header-subtitle {
    color: #CBD5E1;
  }
  
  .chatbot-card {
    background: #111827;
    box-shadow: 0 10px 40px rgba(15, 23, 42, 0.5);
  }
  
  .chatbot-card:hover {
    box-shadow: 0 20px 60px rgba(37, 99, 235, 0.3);
  }
  
  .card-title {
    color: #F8FAFC;
  }
  
  .card-description {
    color: #CBD5E1;
  }
  
  .card-button.disabled {
    background: #1E293B;
    color: #64748B;
  }
  
  .footer-text {
    color: #64748B;
  }
}

/* Responsive */
@media (max-width: 768px) {
  .main-container {
    padding: 20px 16px;
  }
  
  .header-title {
    font-size: 2rem;
  }
  
  .header-subtitle {
    font-size: 1rem;
  }
  
  .chatbot-card {
    padding: 24px;
  }
  
  .cards-container {
    grid-template-columns: 1fr;
  }
}
</style>
""", unsafe_allow_html=True)

# Main content
st.markdown("""
<div class='main-container'>
  <div class='header-section'>
    <h1 class='header-title'>📦 Smart Logistics Assistant</h1>
    <p class='header-subtitle'>Choose an assistant to start chatting</p>
  </div>
</div>
""", unsafe_allow_html=True)

# Chatbot cards
col1, col2 = st.columns(2, gap="medium")

with col1:
    st.markdown("""
    <div class='chatbot-card'>
        <div class='card-icon'>🤖</div>
        <div class='card-title'>LogiBot</div>
        <div class='card-description'>AI Logistics Customer Support</div>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("Start Chat", key="chat_btn_1", use_container_width=True, help="Open chat with LogiBot"):
        st.switch_page("pages/1_Chat.py")

with col2:
    st.markdown("""
    <div class='chatbot-card'>
        <div class='card-icon'>🧑</div>
        <div class='card-title'>KZCHATBOX</div>
        <div class='card-description'>
            Smart Customer Support Assistant
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button(
        "Start Chat",
        key="btn_kzchatbox_page",
        use_container_width=True
    ):
        st.switch_page("pages/KZCHATBOT.py")


st.markdown("""
<div class='footer-text'>
  <p>🚀 Powered by advanced NLP and logistics expertise</p>
</div>
""", unsafe_allow_html=True)
