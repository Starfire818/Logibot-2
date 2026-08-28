"""
File: pages/4_Dataset_Analysis.py
Description: Standalone Streamlit page containing dataset-only content:
             total / train / test / n_intents, intent names, samples per intent
             bar chart, split info, data quality check cards, train/test split tables,
             user feedback summary, plus a robustness CSV link to Model Evaluation.
"""

import copy
import os

import pandas as pd
import streamlit as st
from chatbot import LogisticsChatbot
from nlp_evaluation import (
    HOLDOUT_PRED_PATH,
    QUALITY_PATH,
    RESPONSE_EVAL_PATH,
    ROBUSTNESS_CASES,
    ROBUSTNESS_PATH,
    TEST_SPLIT_PATH,
    TRAIN_SPLIT_PATH,
    USER_FEEDBACK_PATH,
    intent_correct,
    load_json,
)

st.set_page_config(
    page_title="LogiBot - Dataset & Response Analysis",
    page_icon="📁",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📁 Dataset Analysis — LogiBot")
st.caption(
    "Dataset composition, intent distribution, train/test split, data quality checks, "
    "response-generation evaluation, structured robustness probes with recorded responses, "
    "and user satisfaction summary from live chat ratings."
)

st.sidebar.markdown("### Quick navigation")
if st.sidebar.button("← Home / Start chat"):
    st.switch_page("app.py")
if st.sidebar.button("💬 Chat"):
    st.switch_page("pages/1_Chat.py")
if st.sidebar.button("🔍 NLP Analysis"):
    st.switch_page("pages/2_NLP_Analysis.py")
if st.sidebar.button("📊 Model Evaluation"):
    st.switch_page("pages/3_Model_Evaluation.py")
if st.sidebar.button("📘 About LogiBot"):
    st.switch_page("pages/5_About_LogiBot.py")

quality = load_json(QUALITY_PATH, {})
response_eval = load_json(RESPONSE_EVAL_PATH, {})

st.subheader("Dataset overview")
samples_per = quality.get("samples_per_intent") or {}
total = quality.get("rows_after_exact_duplicate_drop_for_split", 0)
if os.path.exists(TRAIN_SPLIT_PATH):
    train_n = len(pd.read_csv(TRAIN_SPLIT_PATH))
else:
    train_n = 0
if os.path.exists(TEST_SPLIT_PATH):
    test_n = len(pd.read_csv(TEST_SPLIT_PATH))
else:
    test_n = 0
n_intents = quality.get("n_intents", len(samples_per))

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total samples", total or quality.get("total_rows", "n/a"))
c2.metric("Train samples", train_n or "n/a")
c3.metric("Test samples", test_n or "n/a")
c4.metric("#Intents", n_intents or "n/a")
st.caption(f"Train/test split: 80/20 stratified, random_state=42 (duplicates of exact (intent, utterance) pairs dropped only for splitting to avoid leakage).")

with st.expander("Intent names ({})".format(n_intents or 0)):
    if samples_per:
        names_df = pd.DataFrame(
            [{"intent": k, "samples": v} for k, v in sorted(samples_per.items(), key=lambda kv: kv[1], reverse=True)]
        )
        st.dataframe(names_df, use_container_width=True, hide_index=True)
    else:
        st.caption("Run train_model.py to populate intent-name counts.")

if samples_per:
    st.subheader("Intent distribution (samples per intent)")
    st.bar_chart(pd.Series(samples_per))

st.divider()
st.subheader("Data quality check (no rows auto-deleted)")
q1, q2, q3, q4 = st.columns(4)
q1.metric("Missing utterances", quality.get("missing_utterances", 0))
q2.metric("Missing intent labels", quality.get("missing_intent_labels", 0))
q3.metric("Empty text rows", quality.get("empty_text", 0))
q4.metric("Duplicate rows", quality.get("duplicate_rows", 0))
q5, q6 = st.columns(2)
q5.metric("Class imbalance (max/min)", quality.get("class_imbalance_ratio_max_over_min", "n/a"))
leakage = quality.get("train_test_leakage", {})
q6.metric("Train/test leakage (overlaps)", leakage.get("overlapping_utterances", "n/a"))
st.caption(quality.get("class_imbalance_note", ""))
st.caption(quality.get("train_test_leakage", {}).get("note", ""))
if quality.get("actions_taken"):
    with st.expander("Data-preparation actions taken"):
        for line in quality.get("actions_taken"):
            st.write(f"- {line}")

st.divider()
st.subheader("Response generation evaluation")
re_method = response_eval.get("method", "template_and_database_mapping")
st.info(
    f"**Actual response approach:** {re_method}\n\n"
    "LogiBot answers come from three deterministic sources, checked in this order:\n"
    "1. **Database (orders.csv) lookup + natural-language builder** — Tracking / ETA / Expedite / Order-summary (actionable intents).\n"
    "2. **Replies.csv Cosine-similar match** — closest matched dataset reply reused, lightly contextualized.\n"
    "3. **Random handwritten template from intents.json** — every other intent; ≥ 2 strings each so answers vary.\n\n"
    "Responses are NOT produced by a generative model."
)
st.write(response_eval.get("bleu_rouge_reason", ""))
st.write(f"**BLEU/ROUGE used:** {response_eval.get('bleu_rouge_used')}")
st.write(f"**Template coverage (intents with ≥ 1 response):** {100 * float(response_eval.get('template_coverage') or 0):.0f}%")
missing = response_eval.get("intents_missing_templates") or []
if missing:
    st.warning("Intents without a template: " + ", ".join(missing))
else:
    st.success("Every training intent has at least one mapped response template.")
if response_eval.get("relevance_evaluated"):
    r1, r2 = st.columns(2)
    r1.metric("Robustness probes checked for response relevance", response_eval.get("relevance_total", 0))
    r2.metric(
        "Replies relevant (intent correct AND not unknown fallback)",
        "{} / {}  ({:.0f}%)".format(
            response_eval.get("relevance_relevant", 0),
            response_eval.get("relevance_total", 1),
            100 * float(response_eval.get("relevance_rate") or 0),
        ),
    )

st.divider()
st.subheader("Robustness probes (intent recognition + response records)")
st.caption(
    "These 55 hand-designed cases exercise all 13 logistics intents and cover "
    "§15 categories: Tracking, delivery status, ETA, address change, missing parcel, "
    "damaged parcel, returns, shipping cost, pickup, international shipping, payment, "
    "complaint, general questions, typos, unclear input, plus multiple wordings of the "
    "same intent. Each row records: User Input / Expected / Predicted / Correct / Response / "
    "Response-unknown / Confidence / Fallback Used / Fallback Method / NLP Method / Similarity / "
    "Case Type / Preprocessed."
)
if os.path.exists(ROBUSTNESS_PATH):
    robust = pd.read_csv(ROBUSTNESS_PATH)
    if "correct" in robust.columns:
        rb1, rb2 = st.columns(2)
        rb1.metric("Robustness correct", f"{int(robust['correct'].sum())} / {len(robust)}")
        rb2.metric(
            "Robustness accuracy",
            "{:.1f}%".format(100 * int(robust["correct"].sum()) / max(len(robust), 1)),
        )
    if st.button("🚀 Re-run all 55 robustness probes now (in browser)", type="primary"):
        with st.spinner("Evaluating layered pipeline on 55 hand-designed probes…"):
            chatbot = LogisticsChatbot()
            rows = []
            unknown_hint = "I'm not quite sure how to help with that"
            for text, expected, case_type in ROBUSTNESS_CASES:
                chatbot.reset_session_context()
                reply, analysis = chatbot.get_bot_response_with_analysis(text)
                predicted = str(analysis.get("detected_intent") or "")
                correct = bool(intent_correct(predicted, expected))
                is_unknown = isinstance(reply, str) and unknown_hint.lower() in str(reply).lower()
                rows.append({
                    "user_input": text,
                    "expected_intent": expected,
                    "predicted_intent": predicted,
                    "correct": correct,
                    "response": reply,
                    "response_is_unknown_fallback": is_unknown,
                    "decision_confidence": analysis.get("decision_confidence"),
                    "fallback_used": bool(analysis.get("fallback_used")),
                    "fallback_method": analysis.get("fallback_method"),
                    "nlp_method": analysis.get("nlp_method"),
                    "similarity_score": analysis.get("similarity_score"),
                    "case_type": case_type,
                    "preprocessed": analysis.get("preprocessed_text"),
                })
            fresh = pd.DataFrame(rows)
            fresh.to_csv(ROBUSTNESS_PATH, index=False)
            st.success(f"Robustness: {int(fresh['correct'].sum())} / {len(fresh)} correct")
            st.rerun()
    st.dataframe(robust, use_container_width=True)
    with st.expander("Add your own custom robustness test"):
        custom_input = st.text_input("User input (the sentence to classify)", key="ds_custom_input")
        labels = list(quality.get("samples_per_intent", {}).keys()) or []
        if not labels and os.path.exists(HOLDOUT_PRED_PATH):
            h = pd.read_csv(HOLDOUT_PRED_PATH)
            labels = sorted(set(h["expected_intent"].tolist()) | set(h["predicted_intent"].tolist()))
        custom_expected = st.selectbox("Expected intent", labels or ["tracking_request"], key="ds_custom_expected")
        custom_type = st.selectbox(
            "Case type",
            ["paraphrase", "typo", "short", "long", "unclear", "unknown", "chinese", "custom"],
            key="ds_custom_type",
        )
        if st.button("Run this test", key="ds_custom_run") and custom_input.strip():
            chatbot = LogisticsChatbot()
            reply, analysis = chatbot.get_bot_response_with_analysis(custom_input.strip())
            predicted = str(analysis.get("detected_intent") or "")
            correct = bool(intent_correct(predicted, custom_expected))
            st.write(f"**Predicted intent:** {predicted}")
            st.write(f"**Correct?** {'✅ Yes' if correct else '❌ No'}")
            st.write(f"**NLP method:** {analysis.get('nlp_method')}")
            st.write(f"**Decision confidence:** {analysis.get('decision_confidence')}")
            st.write(f"**Fallback used:** {analysis.get('fallback_used')} — {analysis.get('fallback_method')}")
            st.write(f"**Bot reply:** {reply}")

st.divider()
st.subheader("User satisfaction (live ratings from Chat page)")
st.caption("Anonymous 1–5 star ratings + helpfulness yes/no + free-text comment are collected in pages/1_Chat.py after every answer. Demo: run the chat, ask a few questions, then return here to see ratings.")
if os.path.exists(USER_FEEDBACK_PATH):
    fb = pd.read_csv(USER_FEEDBACK_PATH)
    if len(fb) > 0:
        f1, f2, f3 = st.columns(3)
        f1.metric("#Responses rated", len(fb))
        avg = round(float(fb["rating_1_5"].dropna().mean()), 2) if "rating_1_5" in fb.columns else 0
        f2.metric("Average rating (1–5)", avg if avg else "n/a")
        if "helpful_bool" in fb.columns:
            yes = int((fb["helpful_bool"].astype(str).str.lower() == "yes").sum())
            answered = int(fb["helpful_bool"].astype(str).isin({"yes", "no"}).sum())
            pct = round(100 * yes / max(answered, 1), 1)
            f3.metric("%Helpful (Yes/No responses)", f"{pct}% ({yes}/{answered})" if answered else "n/a")
        st.dataframe(fb.tail(25), use_container_width=True)
    else:
        st.info("No user ratings yet. Try the chat page → pick stars → Submit feedback, then refresh.")
else:
    st.info("No user feedback file yet. The CSV is created automatically on the first submitted rating.")

st.divider()
st.subheader("Train / Test splits")
sc1, sc2 = st.columns(2)
with sc1:
    st.markdown("#### Train (80%)")
    if os.path.exists(TRAIN_SPLIT_PATH):
        st.dataframe(pd.read_csv(TRAIN_SPLIT_PATH), use_container_width=True)
with sc2:
    st.markdown("#### Test (20%)")
    if os.path.exists(TEST_SPLIT_PATH):
        st.dataframe(pd.read_csv(TEST_SPLIT_PATH), use_container_width=True)
