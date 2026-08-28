"""
NLP Analysis page for LogiBot.

Shows the latest chatbot NLP decision and a history table for assignment demos.
"""

import copy
import os

import pandas as pd
import streamlit as st

from chatbot import LogisticsChatbot

st.set_page_config(page_title="LogiBot NLP Analysis", page_icon="🧠", layout="wide")

st.title("NLP Analysis")
st.caption("Demonstration view of LogiBot intent classification. This is not generative AI.")

if "messages" not in st.session_state:
    st.session_state.messages = []


def _chatbot_cache_key():
    paths = ["chatbot.py", "nlp_preprocess.py", "model/chatbot_model.pkl", "model/vectorizer.pkl", "data/orders.csv"]
    return tuple(os.path.getmtime(path) if os.path.exists(path) else 0 for path in paths)


@st.cache_resource
def initialize_chatbot(cache_key):
    return LogisticsChatbot(confidence_threshold=0.30)


try:
    chatbot = initialize_chatbot(_chatbot_cache_key())
except Exception as exc:
    st.error(f"Could not load LogiBot. Run `python train_model.py` first. Details: {exc}")
    st.stop()

if "logibot_context" in st.session_state:
    chatbot.session_context = copy.deepcopy(st.session_state.logibot_context)

st.markdown("""
LogiBot classifies user text with **TF-IDF + LinearSVC**.  
Supporting methods: **cosine similarity** (fallback) and **keyword / regex** (tracking IDs, ETA, pickup, etc.).  
Replies come from **templates**, **similar dataset replies**, or the **orders database** — not from a language model.
""")

with st.expander("NLP Pipeline diagram (for presentation)"):
    st.markdown("""
```
User Input
   │
   ▼
Text Cleaning (NFKC Unicode, lowercase, strip URLs/HTML, keep [a-z0-9 CJK '-])
   │
   ▼
Tokenization  (NLTK word_tokenize; Chinese phrases kept whole)
   │
   ▼
Stop-word Removal (domain-safe list: articles/copulas/fillers removed,
│                  WH-words (where/when/what/how) KEPT to distinguish
│                  Tracking vs ETA)
▼
Lemmatization (WordNet Lemmatizer, verb→base then noun→base; no stemming)
│
▼
┌─────────────────────────────────────────────────────────┐
│  LAYERED INTENT CLASSIFICATION (priority order)        │
│                                                         │
│  1. Rule-based NLP  ──► explicit regex + entity        │
│     (highest priority)    detection, affirmation,       │
│                          bare-ID message, ETA/tracking  │
│                          regex, Address Change,         │
│                          International country check    │
│                          → dispatch if matched.         │
│                                                         │
│  2. TF-IDF + LinearSVC ──► shared preprocessor +        │
│     (main ML model)        tokenizer, ngram_range(1,2), │
│                            max_features=8000, balanced  │
│                            class_weight.                │
│                          → confidence/margin gate.      │
│                                                         │
│  3. Keyword fallback ──► priority 0/0.3/0.5/1/2/3/3.5/4 │
│     (rule-based weaker)   keyword map + regex blocks.  │
│                          → dispatch if matched.         │
│                                                         │
│  4. Cosine Similarity ──► per-intent labelled training │
│     (fallback if ML         examples using the SAME     │
│         uncertain)         TF-IDF vectorizer.           │
│                          → similarity_threshold 0.22.   │
│                          → disambiguate Typos/Weird.    │
│                                                         │
│  5. Unknown / default template fallback.               │
└─────────────────────────────────────────────────────────┘
   │
   ▼
Response Generation (3-tier, NO generative AI):
  1. Database lookup + human-readable builder
     (Tracking / ETA / Expedite / Order-summary)
  2. replies.csv Cosine match → matched answer reused
  3. Random handwritten template (≥ 2 per intent, from intents.json)
   │
   ▼
Reply text + NLP analysis trace (intent, method, conf,
                                  fallback, similarity, preprocessed text)
```
""")

probe = st.text_input("Try a sentence", placeholder="Where is my parcel?")
if st.button("Analyse", type="primary") and probe.strip():
    if "logibot_context" in st.session_state:
        chatbot.session_context = copy.deepcopy(st.session_state.logibot_context)
    reply, analysis = chatbot.get_bot_response_with_analysis(probe.strip())
    st.session_state.logibot_context = copy.deepcopy(chatbot.session_context)
    st.session_state.messages.append({"role": "user", "content": probe.strip()})
    st.session_state.messages.append({
        "role": "assistant",
        "content": reply,
        "metadata": {
            "intent": analysis.get("detected_intent"),
            "conf": analysis.get("decision_confidence", 0.0),
            "nlp_method": analysis.get("nlp_method"),
            "fallback_used": analysis.get("fallback_used"),
            "fallback_method": analysis.get("fallback_method"),
            "similarity_score": analysis.get("similarity_score"),
            "preprocessed_text": analysis.get("preprocessed_text"),
            "ml_intent": analysis.get("ml_intent"),
            "ml_confidence": analysis.get("ml_confidence"),
            "decision_margin": analysis.get("decision_margin"),
        },
    })

last_assistant = next(
    (msg for msg in reversed(st.session_state.messages) if msg.get("role") == "assistant" and msg.get("metadata")),
    None,
)

if last_assistant:
    meta = last_assistant["metadata"]
    conf = meta.get("conf") or 0.0
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Detected Intent")
        st.write(meta.get("intent", "n/a"))
        st.subheader("NLP Method")
        st.write(meta.get("nlp_method") or "TF-IDF + LinearSVC")
        st.subheader("Decision / Confidence")
        st.write(f"{conf:.0%}")
    with col2:
        st.subheader("Fallback Used")
        st.write("Yes" if meta.get("fallback_used") else "No")
        if meta.get("fallback_used"):
            st.subheader("Fallback Method")
            st.write(meta.get("fallback_method") or "n/a")
            if meta.get("similarity_score") is not None:
                st.subheader("Similarity Score")
                st.write(f"{float(meta['similarity_score']):.0%}")
        if meta.get("preprocessed_text"):
            st.subheader("Preprocessed Text")
            st.write(meta.get("preprocessed_text"))
    st.subheader("Bot reply")
    st.info(last_assistant.get("content", ""))
else:
    st.info("Send a message on the Chat page, or analyse a sentence above.")

rows = []
for msg in st.session_state.messages:
    if msg.get("role") != "assistant" or not msg.get("metadata"):
        continue
    meta = msg["metadata"]
    rows.append({
        "Detected Intent": meta.get("intent"),
        "NLP Method": meta.get("nlp_method"),
        "Decision/Confidence": meta.get("conf"),
        "Fallback Used": "Yes" if meta.get("fallback_used") else "No",
        "Fallback Method": meta.get("fallback_method"),
        "Similarity Score": meta.get("similarity_score"),
        "Reply": msg.get("content"),
    })

if rows:
    st.subheader("Session history")
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
