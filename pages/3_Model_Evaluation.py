"""
File: pages/3_Model_Evaluation.py
Description: Standalone Streamlit page containing ONLY model-evaluation content:
             accuracy / precision / recall / F1, classification report,
             confusion matrix CSV + PNG, model comparison table (LinearSVC vs LogReg),
             and per-row holdout predictions.
             Separated from the dataset page to match the assignment suggested 5-page layout.
"""

import copy
import os

import pandas as pd
import streamlit as st
from chatbot import LogisticsChatbot
from nlp_evaluation import (
    CONFUSION_CSV_PATH,
    CONFUSION_PNG_PATH,
    HOLDOUT_PRED_PATH,
    METRICS_PATH,
    MODEL_COMPARISON_JSON,
    MODEL_COMPARISON_PATH,
    ROBUSTNESS_CASES,
    ROBUSTNESS_PATH,
    load_json,
)

st.set_page_config(
    page_title="LogiBot - Model Evaluation",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📊 Model Evaluation — LogiBot NLP")
st.caption(
    "Pure ML holdout evaluation (80/20 stratified split, same TF-IDF vectorizer) and "
    "an end-to-end layered-pipeline robustness test. Screenshots here can be pasted into "
    "the report §4 Results and Discussion."
)

st.sidebar.markdown("### Quick navigation")
if st.sidebar.button("← Home / Start chat"):
    st.switch_page("app.py")
if st.sidebar.button("💬 Chat"):
    st.switch_page("pages/1_Chat.py")
if st.sidebar.button("🔍 NLP Analysis"):
    st.switch_page("pages/2_NLP_Analysis.py")
if st.sidebar.button("📁 Dataset Analysis"):
    st.switch_page("pages/4_Dataset_Analysis.py")
if st.sidebar.button("📘 About LogiBot"):
    st.switch_page("pages/5_About_LogiBot.py")

metrics = load_json(METRICS_PATH, {})
comparison = load_json(MODEL_COMPARISON_JSON, {}).get("comparison") or []

if not metrics:
    st.warning("No evaluation artifacts yet. Run `python train_model.py` on the command line to generate the model and reports.")
    st.stop()

st.subheader("Headline ML performance (LinearSVC, holdout 35 rows)")
mc1, mc2, mc3, mc4 = st.columns(4)
mc1.metric("Accuracy", f"{metrics.get('accuracy', 0):.1%}")
mc2.metric("Precision (weighted)", f"{metrics.get('precision_weighted', 0):.1%}")
mc3.metric("Recall (weighted)", f"{metrics.get('recall_weighted', 0):.1%}")
mc4.metric("F1 (weighted)", f"{metrics.get('f1_weighted', 0):.1%}")
st.caption(
    "Weighted metrics are the correct headline for this imbalanced 18-class dataset "
    "(min class size 1–3 test rows)."
)

st.subheader("Macro-average (unweighted)")
ma1, ma2, ma3, ma4 = st.columns(4)
ma1.metric("Precision (macro)", f"{metrics.get('precision_macro', 0):.1%}")
ma2.metric("Recall (macro)", f"{metrics.get('recall_macro', 0):.1%}")
ma3.metric("F1 (macro)", f"{metrics.get('f1_macro', 0):.1%}")
ma4.metric("#Intents", metrics.get("n_intents", 0))

st.subheader("Classification report (per label)")
cr = metrics.get("classification_report") or {}
cr_rows = []
for label, box in cr.items():
    if label in {"accuracy", "macro avg", "weighted avg"}:
        continue
    if isinstance(box, dict):
        cr_rows.append({
            "intent": label,
            "precision": box.get("precision"),
            "recall": box.get("recall"),
            "f1": box.get("f1-score"),
            "support": int(box.get("support", 0)),
        })
if cr_rows:
    st.dataframe(pd.DataFrame(cr_rows), use_container_width=True, hide_index=True)

st.divider()
st.subheader("📈 Model comparison (assignment §14: different-solution baseline)")
st.info(
    "The deployed LogiBot still uses **TF-IDF + LinearSVC** as its main model. The "
    "TF-IDF + Logistic Regression row is a baseline comparison *only*, to satisfy the "
    "assignment requirement that each group member implements a different chatbot solution. "
    "It is NOT wired into the live chat response pipeline."
)
if os.path.exists(MODEL_COMPARISON_PATH):
    cdf = pd.read_csv(MODEL_COMPARISON_PATH)
    st.dataframe(cdf, use_container_width=True, hide_index=True)
    chart_df = cdf.set_index("model")[
        ["accuracy", "precision_weighted", "recall_weighted", "f1_weighted"]
    ].T.astype(float)
    st.bar_chart(chart_df, use_container_width=True)
    winner = metrics.get("comparison_winner_weighted_f1")
    if winner:
        st.success(f"Best weighted F1 on this dataset and split: **{winner}**.")
else:
    st.caption("model_comparison.csv is regenerated the next time you run train_model.py.")

st.divider()
st.subheader("Confusion matrix (Actual Intent vs Predicted Intent)")
cm_png = metrics.get("confusion_matrix_png") or CONFUSION_PNG_PATH
if os.path.exists(cm_png):
    st.image(cm_png, caption="Confusion matrix: Actual vs Predicted (holdout test set, 18 labels)", use_container_width=True)
cm_csv = metrics.get("confusion_matrix_csv") or CONFUSION_CSV_PATH
if os.path.exists(cm_csv):
    with st.expander("Confusion matrix CSV (raw values)"):
        st.dataframe(pd.read_csv(cm_csv), use_container_width=True)

st.subheader("Holdout predictions (per row, with decision confidence + margin)")
if os.path.exists(HOLDOUT_PRED_PATH):
    hp = pd.read_csv(HOLDOUT_PRED_PATH)
    hp_ok = hp["correct"].sum() if "correct" in hp.columns else None
    if hp_ok is not None:
        st.caption(f"Rows correct by pure LinearSVC prediction: {int(hp_ok)} / {len(hp)} ({100*hp_ok/max(len(hp),1):.1f}%)")
    st.dataframe(hp, use_container_width=True)

st.divider()
st.subheader("End-to-end robustness (layered live pipeline)")
st.caption(
    "Pure ML holdout accuracy is capped low by the tiny 18-class dataset, so we also "
    "exercise the real layered pipeline: rule-based first → LinearSVC with confidence/margin "
    "gate → keyword fallback → cosine similarity fallback. These results are the "
    "presentable correctness metric for the demo."
)
if os.path.exists(ROBUSTNESS_PATH):
    robust = pd.read_csv(ROBUSTNESS_PATH)
    if "correct" in robust.columns:
        n = int(robust["correct"].sum())
        d = len(robust)
        rc1, rc2, rc3 = st.columns(3)
        rc1.metric("Robustness correct", f"{n} / {d}")
        rc2.metric("Robustness accuracy", f"{100*n/max(d,1):.1f}%")
        type_counts = (
            robust.assign(ok=robust["correct"])
            .groupby("case_type")["ok"]
            .agg(["sum", "count"])
            .rename(columns={"sum": "correct", "count": "total"})
            .reset_index()
        )
        type_counts["accuracy"] = (type_counts["correct"] / type_counts["total"]).round(3)
        rc3.metric("Case types covered", len(type_counts))
        st.dataframe(type_counts, use_container_width=True, hide_index=True)
    if st.button("Re-run 55 robustness probes now (in browser)", type="secondary"):
        with st.spinner("Evaluating layered pipeline on 55 hand-designed probes…"):
            chatbot = LogisticsChatbot()
            rows = []
            unknown_hint = "I'm not quite sure how to help with that"
            for text, expected, case_type in ROBUSTNESS_CASES:
                chatbot.reset_session_context()
                reply, predicted, confidence = chatbot.get_bot_response(text)
                analysis = chatbot.last_nlp_analysis or {}
                predicted_name = str(predicted)
                correct = predicted_name == expected
                if expected in {"unknown", "unclear"}:
                    correct = predicted_name in {"unknown_fallback", "Weird", "Typos", "General", "default"}
                is_unknown = isinstance(reply, str) and unknown_hint.lower() in str(reply).lower()
                rows.append({
                    "user_input": text,
                    "expected_intent": expected,
                    "predicted_intent": predicted,
                    "correct": correct,
                    "response": reply,
                    "response_is_unknown_fallback": is_unknown,
                    "decision_confidence": round(float(confidence or 0), 4),
                    "fallback_used": bool(analysis.get("fallback_used")),
                    "fallback_method": analysis.get("fallback_method"),
                    "nlp_method": analysis.get("nlp_method"),
                    "similarity_score": analysis.get("similarity_score"),
                    "case_type": case_type,
                })
            fresh = pd.DataFrame(rows)
            fresh.to_csv(ROBUSTNESS_PATH, index=False)
            st.success(f"Done — correct: {int(fresh['correct'].sum())} / {len(fresh)}")
            st.rerun()
    st.dataframe(robust, use_container_width=True)
    with st.expander("Add your own custom robustness test"):
        custom_input = st.text_input("User input", key="me_custom_input")
        expected_options = metrics.get("labels") or [] or ["tracking_request"]
        custom_expected = st.selectbox("Expected intent", sorted(set(expected_options)), key="me_custom_expected")
        custom_type = st.selectbox("Case type", ["paraphrase", "typo", "short", "long", "unclear", "unknown", "chinese", "custom"], key="me_custom_type")
        if st.button("Run this test", key="me_custom_run") and custom_input.strip():
            chatbot = LogisticsChatbot()
            reply, predicted, confidence = chatbot.get_bot_response(custom_input.strip())
            analysis = chatbot.last_nlp_analysis or {}
            correct = str(predicted) == str(custom_expected)
            st1, st2 = st.columns(2)
            st1.write(f"**Predicted intent:** {predicted}")
            st1.write(f"**Correct?** {'✅' if correct else '❌'}")
            st1.write(f"**Decision confidence:** {float(confidence or 0):.1%}")
            st2.write(f"**NLP method:** {analysis.get('nlp_method')}")
            st2.write(f"**Fallback used:** {analysis.get('fallback_used')} — {analysis.get('fallback_method')}")
            st.write(f"**Bot reply:** {reply}")
