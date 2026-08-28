"""
File: pages/1_Chat.py
Description: ChatGPT-style chat page for interacting with LogiBot.
             Displays chat history with proper message bubbles and scrolling.
"""

import copy
import os

import streamlit as st
from chatbot import LogisticsChatbot

# (packaged-build feedback preview marker)
# Persistence status + recent feedback preview (packaged build only).
try:
    from chatbot import _persistence_status as _pstat
    _s = _pstat()
    with st.sidebar.expander("\U0001F4BE Feedback Persistence", expanded=True):
        _tiers = ["\u2705 Local CSV (session only)"]
        if _s.get("gsheets_configured"):
            _tiers.append("\U0001F7E9 Google Sheets (permanent)")
        else:
            _tiers.append("\u26AA Google Sheets (not configured)")
        if _s.get("github_configured"):
            _tiers.append("\U0001F7E5 GitHub CSV commit (permanent)")
        else:
            _tiers.append("\u26AA GitHub CSV commit (not configured)")
        for _t in _tiers:
            st.markdown(f"- {_t}")
        if not _s.get("gsheets_configured") and not _s.get("github_configured"):
            st.caption("Local-only persistence: rows will be LOST on Streamlit Cloud restart. Configure one of the two remote backends above in the app's Secrets settings to make feedback permanent.")
        else:
            st.caption("Rows written locally AND mirrored to the cloud — survive Streamlit restarts.")
    # Preview the 10 most recent feedback rows (reads from local CSV which
    # is the authoritatively available store within this container).
    import os as _os
    _fb_path = "data/user_feedback.csv"
    try:
        from chatbot import _resolve_path as _rp; _fb_path = _rp(_fb_path)
    except Exception:  # noqa: BLE001
        pass
    if _os.path.isfile(_fb_path):
        import pandas as _pd
        try:
            _df = _pd.read_csv(_fb_path)
            if not _df.empty:
                _count = min(10, len(_df))
                _recent = _df.tail(_count).iloc[::-1].reset_index(drop=True)
                with st.sidebar.expander(f"\U0001F4CB Recent feedback (last {_count})", expanded=False):
                    _cols_subset = [c for c in ["timestamp_utc","rating_1_5","helpful_bool","detected_intent","comment","user_message"] if c in _recent.columns]
                    st.dataframe(_recent[_cols_subset] if _cols_subset else _recent, use_container_width=True, height=260)
        except Exception:  # noqa: BLE001
            pass
except Exception:  # noqa: BLE001
    pass



# Page configuration
st.set_page_config(
    page_title="LogiBot - Smart Logistics Assistant",
    page_icon="🤖",
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
  background: linear-gradient(180deg, #F0F4FF 0%, #FFFFFF 100%) !important;
}

#MainMenu, footer {
  visibility: hidden;
}

/* Full-height layout */
.stApp {
  display: flex;
  flex-direction: column;
  height: 100vh;
}

.main {
  display: flex;
  flex-direction: column;
  height: 100vh;
  padding: 0 !important;
  overflow: hidden;
}

.block-container {
  padding: 0 !important;
  display: flex;
  flex-direction: column;
  height: 100vh;
}

/* Header */
.chat-header {
  flex-shrink: 0;
  background: linear-gradient(135deg, #2563EB 0%, #60A5FA 100%);
  color: white;
  padding: 16px 24px;
  display: flex;
  align-items: center;
  gap: 16px;
  box-shadow: 0 4px 12px rgba(37, 99, 235, 0.15);
}

.back-button {
  background: rgba(255, 255, 255, 0.2);
  border: none;
  color: white;
  width: 40px;
  height: 40px;
  border-radius: 8px;
  font-size: 20px;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.2s ease;
}

.back-button:hover {
  background: rgba(255, 255, 255, 0.3);
  transform: translateX(-2px);
}

.back-button:active {
  transform: translateX(0);
}

.header-text h1 {
  margin: 0;
  font-size: 1.3rem;
  font-weight: 700;
}

.header-text p {
  margin: 2px 0 0 0;
  font-size: 0.85rem;
  opacity: 0.9;
}

/* Chat container */
.chat-container {
  flex: 1;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  background: #F8FAFC;
}

.chat-messages {
  flex: 1;
  overflow-y: auto;
  padding: 24px;
  display: flex;
  flex-direction: column;
  gap: 12px;
  scroll-behavior: smooth;
}

/* Custom scrollbar */
.chat-messages::-webkit-scrollbar {
  width: 8px;
}

.chat-messages::-webkit-scrollbar-track {
  background: transparent;
}

.chat-messages::-webkit-scrollbar-thumb {
  background: rgba(148, 163, 184, 0.3);
  border-radius: 4px;
}

.chat-messages::-webkit-scrollbar-thumb:hover {
  background: rgba(148, 163, 184, 0.5);
}

/* Empty state */
.empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 100%;
  color: #94A3B8;
  text-align: center;
}

.empty-state-icon {
  font-size: 4rem;
  margin-bottom: 16px;
  opacity: 0.3;
}

.empty-state-title {
  font-size: 1.2rem;
  font-weight: 600;
  color: #475569;
  margin-bottom: 8px;
}

.empty-state-text {
  font-size: 0.95rem;
  color: #94A3B8;
}

/* Message wrapper */
.message-group {
  display: flex;
  margin: 0;
  padding: 0;
}

.message-group.user {
  justify-content: flex-end;
}

.message-group.assistant {
  justify-content: flex-start;
}

/* Message bubble */
.message-bubble {
  max-width: 65%;
  word-wrap: break-word;
  line-height: 1.5;
  font-size: 0.95rem;
  padding: 12px 16px;
  border-radius: 16px;
  animation: slideIn 0.3s ease-out;
}

@keyframes slideIn {
  from {
    opacity: 0;
    transform: translateY(8px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

.message-bubble.user {
  background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%);
  color: white;
  border-bottom-right-radius: 4px;
  box-shadow: 0 2px 8px rgba(37, 99, 235, 0.2);
}

.message-bubble.assistant {
  background: white;
  color: #0F172A;
  border-bottom-left-radius: 4px;
  box-shadow: 0 2px 8px rgba(15, 23, 42, 0.08);
  border: 1px solid #E2E8F0;
}

.message-metadata {
  font-size: 0.75rem;
  color: #64748B;
  margin-top: 4px;
  opacity: 0.7;
}

/* Input area */
.input-section {
  flex-shrink: 0;
  background: white;
  border-top: 1px solid #E2E8F0;
  padding: 16px 24px 24px;
}

.input-wrapper {
  display: flex;
  gap: 12px;
  align-items: flex-end;
}

.text-input-container {
  flex: 1;
  display: flex;
  flex-direction: column;
}

.input-box {
  width: 100%;
  background: white;
  border: 1.5px solid #E2E8F0;
  border-radius: 20px;
  padding: 12px 18px;
  font-size: 0.95rem;
  font-family: inherit;
  resize: none;
  transition: all 0.2s ease;
}

.input-box:focus {
  outline: none;
  border-color: #2563EB;
  box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1);
}

.input-box::placeholder {
  color: #94A3B8;
}

.send-btn {
  background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%);
  color: white;
  border: none;
  border-radius: 16px;
  padding: 10px 20px;
  font-weight: 600;
  font-size: 0.9rem;
  cursor: pointer;
  transition: all 0.2s ease;
  height: 42px;
}

.send-btn:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
}

.send-btn:active {
  transform: translateY(0);
}

.clear-btn {
  background: #F1F5F9;
  color: #64748B;
  border: 1px solid #E2E8F0;
  border-radius: 16px;
  padding: 10px 16px;
  font-weight: 500;
  font-size: 0.9rem;
  cursor: pointer;
  transition: all 0.2s ease;
  height: 42px;
}

.clear-btn:hover {
  background: #E2E8F0;
  color: #475569;
}

.clear-btn:active {
  transform: translateY(0);
}

/* Dark Mode */
@media (prefers-color-scheme: dark) {
  body, .stApp, .main, .block-container {
    background: linear-gradient(180deg, #0B1220 0%, #111827 100%) !important;
    color: #F8FAFC !important;
  }
  
  .chat-header {
    background: linear-gradient(135deg, #1D4ED8 0%, #1E40AF 100%);
    box-shadow: 0 4px 12px rgba(15, 23, 42, 0.5);
  }
  
  .chat-container {
    background: #0F172A;
  }
  
  .chat-messages::-webkit-scrollbar-thumb {
    background: rgba(71, 85, 105, 0.4);
  }
  
  .chat-messages::-webkit-scrollbar-thumb:hover {
    background: rgba(71, 85, 105, 0.6);
  }
  
  .empty-state-icon {
    opacity: 0.2;
  }
  
  .empty-state-title {
    color: #CBD5E1;
  }
  
  .empty-state-text {
    color: #64748B;
  }
  
  .message-bubble.assistant {
    background: #1E293B;
    color: #E2E8F0;
    border-color: #334155;
  }
  
  .input-section {
    background: #0F172A;
    border-top-color: #1E293B;
  }
  
  .input-box {
    background: #1E293B;
    border-color: #334155;
    color: #F8FAFC;
  }
  
  .input-box:focus {
    border-color: #2563EB;
    box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.2);
  }
  
  .input-box::placeholder {
    color: #64748B;
  }
  
  .clear-btn {
    background: #1E293B;
    color: #94A3B8;
    border-color: #334155;
  }
  
  .clear-btn:hover {
    background: #334155;
    color: #CBD5E1;
  }
}

/* Responsive */
@media (max-width: 768px) {
  .message-bubble {
    max-width: 85%;
  }
  
  .header-text h1 {
    font-size: 1.1rem;
  }
  
  .input-section {
    padding: 12px 16px 16px;
  }
}
</style>
""", unsafe_allow_html=True)

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = []

def _chatbot_cache_key():
    """Reload the bot when chatbot logic, model artifacts, or order data change."""
    paths = ["chatbot.py", "nlp_preprocess.py", "model/chatbot_model.pkl", "model/vectorizer.pkl", "data/orders.csv"]
    return tuple(os.path.getmtime(path) if os.path.exists(path) else 0 for path in paths)

@st.cache_resource
def initialize_chatbot(cache_key, conf_threshold=0.30):
    return LogisticsChatbot(confidence_threshold=conf_threshold)

try:
    chatbot = initialize_chatbot(_chatbot_cache_key())
except Exception as e:
    st.error(f"❌ Failed to load the AI core pipeline. Details: {e}")
    st.info("💡 Advice: Ensure you have successfully populated data files and run `train_model.py` to compile the artifacts.")
    st.stop()

# Header with back button
col_back, col_header = st.columns([0.8, 5], gap="medium")

with col_back:
    if st.button("← Back", key="back_btn", use_container_width=True):
        st.switch_page("app.py")

with col_header:
    st.markdown("""
    <div class='header-text'>
        <h1>🤖 LogiBot</h1>
        <p>AI Logistics Customer Support</p>
    </div>
    """, unsafe_allow_html=True)

# Chat messages area
st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
chat_placeholder = st.empty()

def render_messages():
    with chat_placeholder.container():
        if len(st.session_state.messages) == 0:
            st.markdown("""
            <div class='empty-state'>
                <div class='empty-state-icon'>💬</div>
                <div class='empty-state-title'>Start a conversation</div>
                <div class='empty-state-text'>Ask about tracking, deliveries, refunds, shipping, or any logistics question</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("<div class='chat-messages'>", unsafe_allow_html=True)
            
            for msg in st.session_state.messages:
                role = msg["role"]
                content = msg["content"]
                
                if role == "user":
                    st.markdown(f"""
                    <div class='message-group user'>
                        <div class='message-bubble user'>{content}</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    metadata_html = ""
                    if "metadata" in msg:
                        intent = msg["metadata"].get("intent", "n/a")
                        conf = msg["metadata"].get("conf", 0.0) or 0.0
                        method = msg["metadata"].get("nlp_method") or "TF-IDF + LinearSVC"
                        fallback = "fallback" if msg["metadata"].get("fallback_used") else "no fallback"
                        metadata_html = (
                            f"<div class='message-metadata'>⚙️ {intent} · {conf:.1%} · {method} · {fallback}</div>"
                        )
                    
                    st.markdown(f"""
                    <div class='message-group assistant'>
                        <div class='message-bubble assistant'>
                            {content}
                            {metadata_html}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)

render_messages()
st.markdown("</div>", unsafe_allow_html=True)

# Input area
st.markdown("<div class='input-section'>", unsafe_allow_html=True)

with st.form(key="chat_form", clear_on_submit=True):
    col_input, col_send, col_clear = st.columns([5, 0.8, 0.8], gap="small")

    with col_input:
        user_text = st.text_area(
            "message",
            key="user_message",
            placeholder="Type your message... (Shift+Enter for new line)",
            height=44,
            label_visibility="collapsed"
        )

    with col_send:
        send_btn = st.form_submit_button("Send", use_container_width=True, help="Send message")

    with col_clear:
        clear_btn = st.form_submit_button("Clear", use_container_width=True, help="Clear chat history")

st.markdown("</div>", unsafe_allow_html=True)

if "logibot_context" in st.session_state:
    chatbot.session_context = copy.deepcopy(st.session_state.logibot_context)
else:
    chatbot.reset_session_context()
    st.session_state.logibot_context = copy.deepcopy(chatbot.session_context)

# Handle message sending
if send_btn and user_text.strip():
    user_message = user_text.strip()
    
    st.session_state.messages.append({
        "role": "user",
        "content": user_message
    })
    
    bot_response, intent, confidence = chatbot.get_bot_response(user_message)
    st.session_state.logibot_context = copy.deepcopy(chatbot.session_context)
    analysis = dict(chatbot.last_nlp_analysis or {})

    st.session_state.messages.append({
        "role": "assistant",
        "content": bot_response,
        "metadata": {
            "intent": intent,
            "conf": confidence,
            "nlp_method": analysis.get("nlp_method"),
            "fallback_used": analysis.get("fallback_used"),
            "fallback_method": analysis.get("fallback_method"),
            "similarity_score": analysis.get("similarity_score"),
            "preprocessed_text": analysis.get("preprocessed_text"),
            "decision_margin": analysis.get("decision_margin"),
        }
    })

    st.rerun()

# Handle clear chat
if clear_btn:
    st.session_state.messages = []
    chatbot.reset_session_context()
    st.session_state.logibot_context = copy.deepcopy(chatbot.session_context)
    st.rerun()

last_assistant = next(
    (msg for msg in reversed(st.session_state.messages) if msg.get("role") == "assistant" and msg.get("metadata")),
    None,
)
if last_assistant:
    meta = last_assistant["metadata"]
    conf = meta.get("conf") or 0.0
    with st.expander("NLP Analysis (last reply)", expanded=True):
        st.write(f"**Detected Intent:** {meta.get('intent', 'n/a')}")
        st.write(f"**NLP Method:** {meta.get('nlp_method') or 'TF-IDF + LinearSVC'}")
        st.write(f"**Decision/Confidence:** {conf:.0%}")
        st.write(f"**Fallback Used:** {'Yes' if meta.get('fallback_used') else 'No'}")
        if meta.get("fallback_used"):
            st.write(f"**Fallback Method:** {meta.get('fallback_method') or 'n/a'}")
            if meta.get("similarity_score") is not None:
                st.write(f"**Similarity Score:** {float(meta.get('similarity_score')):.0%}")
        if meta.get("preprocessed_text"):
            st.caption(f"Preprocessed text: {meta.get('preprocessed_text')}")

    # Simple 1–5 user satisfaction feedback (assignment §17)
    st.markdown("<div class='feedback-section'>", unsafe_allow_html=True)
    feedback_key = "rating_" + str(hash(str(last_assistant.get("content", ""))[:200]))[:8]
    if "feedback_last_rendered_key" not in st.session_state or st.session_state["feedback_last_rendered_key"] != feedback_key:
        st.session_state.pop("fb_helpful", None)
        st.session_state.pop("fb_comment", None)
        st.session_state["feedback_last_rendered_key"] = feedback_key

    st.markdown("**Was LogiBot's last answer helpful?** (Anonymous feedback for assignment evaluation)")
    stars = st.feedback("stars", key=feedback_key + "_stars")
    helpful = st.radio(
        "Helpfulness",
        ["Skip", "Yes — helpful", "No — not helpful"],
        index=0,
        horizontal=True,
        label_visibility="collapsed",
        key=feedback_key + "_helpful",
    )
    comment = st.text_input(
        "Optional comment (sent anonymously with the rating)",
        key=feedback_key + "_comment",
        placeholder="e.g. Wrong intent detected, or: Quick and accurate tracking answer",
        label_visibility="collapsed",
    )
    if st.button("📨 Submit feedback", key=feedback_key + "_submit", type="secondary"):
        helpful_bool = None
        if helpful == "Yes — helpful":
            helpful_bool = True
        elif helpful == "No — not helpful":
            helpful_bool = False
        # stars: st.feedback returns 0-indexed, convert to 1-5
        rating_1_5 = 0
        if stars is not None:
            try:
                rating_1_5 = int(stars) + 1
            except Exception:
                rating_1_5 = 0
        if rating_1_5 < 1 and helpful_bool is None and not comment:
            st.info("Pick a star rating and/or click Yes/No and/or type a comment before submitting.")
        else:
            last_user = next(
                (m["content"] for m in reversed(st.session_state.messages) if m.get("role") == "user"),
                "",
            )
            ok = LogisticsChatbot.save_feedback(
                user_msg=str(last_user),
                bot_reply=str(last_assistant.get("content", "")),
                detected_intent=str(meta.get("intent", "")),
                rating_1_5=rating_1_5 if rating_1_5 >= 1 else 3,
                helpful_bool=helpful_bool,
                comment_str=str(comment).strip() or None,
            )
            if ok:
                st.success(
                    "✅ Thank you! Your rating ({} star{}) has been saved to data/user_feedback.csv.".format(
                        (rating_1_5 if rating_1_5 >= 1 else "n/a"),
                        ("s" if rating_1_5 != 1 else ""),
                    )
                )
            else:
                st.warning("Feedback could not be written right now — the page will still work for chatting.")
    st.markdown("</div>", unsafe_allow_html=True)

# Auto-scroll to latest message
st.components.v1.html("""
<script>
function scrollToBottom() {
  setTimeout(() => {
    const chatMessages = document.querySelector('.chat-messages');
    if (chatMessages) {
      chatMessages.scrollTop = chatMessages.scrollHeight;
    }
  }, 100);
}

scrollToBottom();

const observer = new MutationObserver(scrollToBottom);
const container = document.querySelector('.chat-container');
if (container) {
  observer.observe(container, { childList: true, subtree: true });
}

// Enter sends message; Shift+Enter inserts newline
const textarea = document.querySelector('.input-section textarea');
const sendButton = Array.from(document.querySelectorAll('.input-section button')).find(btn => btn.innerText.trim().toLowerCase() === 'send');
if (textarea && sendButton) {
  textarea.addEventListener('keydown', function(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      sendButton.click();
    }
  });
}
</script>
""", height=0, width=0)
