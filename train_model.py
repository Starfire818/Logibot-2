"""
File: train_model.py
Description: Loads the logistics support dataset, applies the shared NLP
             preprocessor, trains TF-IDF + LinearSVC, writes evaluation
             artifacts, and exports the live chatbot model.
"""

import json
import os
import pickle

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

from nlp_evaluation import (
    CONFUSION_CSV_PATH,
    HOLDOUT_PRED_PATH,
    METRICS_PATH,
    MODEL_COMPARISON_JSON,
    MODEL_COMPARISON_PATH,
    NLP_DIR,
    QUALITY_PATH,
    RESPONSE_EVAL_PATH,
    ROBUSTNESS_CASES,
    ROBUSTNESS_PATH,
    TEST_SPLIT_PATH,
    TRAIN_SPLIT_PATH,
    compute_holdout_metrics,
    evaluate_response_mapping,
    inspect_dataset_quality,
    linear_svc_decision_scores,
    overlap_leakage,
    save_confusion_matrix,
    write_json,
    write_model_comparison,
)
from nlp_preprocess import build_tfidf_vectorizer, preprocess_text


def find_dataset_path():
    """Return the available logistics CSV dataset for training."""
    candidates = [
        "Logistics_Customer_Support_Dataset.csv",
        "Training-dataset-for-chatbots.csv",
        os.path.join("data", "Logistics_Customer_Support_Dataset.csv"),
    ]

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    raise FileNotFoundError(
        "No dataset file was found. Please add Logistics_Customer_Support_Dataset.csv "
        "or the legacy Training-dataset-for-chatbots.csv file to the project root."
    )


def build_response_templates():
    """Create a response dictionary compatible with the current dataset labels."""
    responses = {
        "Tracking": [
            "Thanks for reaching out. I can help you track your shipment. Please share your tracking number or order details so I can check the latest status.",
            "I can help with tracking updates. Please send your tracking number and I will look up the newest shipment status."
        ],
        "Delivery": [
            "I can help with your delivery timeline. Please share your tracking number or order reference, and I will check the expected arrival date.",
            "Thanks for contacting us. I can review your delivery details and help estimate when your parcel should arrive."
        ],
        "Address Change": [
            "I can help with address changes if the parcel has not yet been handed off to the courier. Please send the order number and the correct delivery address.",
            "Address updates are usually possible before dispatch. Please share the order and updated address details so I can check."
        ],
        "Missing Parcel": [
            "I am sorry to hear that. I can help investigate a missing parcel. Please send the tracking number and delivery details so I can review it.",
            "We can investigate a missing shipment. Please share your order number and any delivery information so I can assess the issue."
        ],
        "Damaged Parcel": [
            "I am sorry your parcel arrived damaged. Please send photos of the package and the item, along with your order number, and I will help with the claim process.",
            "Thank you for reporting this. Please provide the order number and photos of the damaged item so I can escalate the case."
        ],
        "Returns": [
            "I can guide you through the return process. Please share your order number and I will explain the available options and refund status.",
            "I can help with return requests and refunds. Please provide the order reference so I can review the next steps."
        ],
        "Shipping Cost": [
            "I can explain the shipping cost. Please share your destination and parcel details, and I will check the fee or any available discount options.",
            "Shipping charges depend on delivery speed, destination, and parcel size. Please share the order details and I can clarify the cost."
        ],
        "Pickup": [
            "I can help with pickup arrangements. Please share your location details and preferred pickup time so I can check availability.",
            "Pickup support is available. Please provide your order reference and preferred time, and I will review the options with you."
        ],
        "International": [
            "I can help with international shipping. Please share the destination and tracking details, and I will explain the delivery timeline and customs information.",
            "International delivery can be affected by customs and transit times. Please share your order reference so I can review the status."
        ],
        "Payment": [
            "I can assist with payment issues. Please share the payment method and order details so I can check why the transaction was rejected or delayed.",
            "Thanks for letting us know. I can help review payment problems and explain available payment methods or next steps."
        ],
        "Business": [
            "Thank you for your interest in our business services. I can help with bulk shipping, API integration, and warehouse support options. Please share your requirements.",
            "I can help with business account and bulk shipping inquiries. Please tell me what type of service you need and I will guide you."
        ],
        "Complaint": [
            "I am sorry to hear this and understand your frustration. Please share your order details so I can review the issue and escalate it properly.",
            "We are sorry for the poor experience. Please provide your order reference and a short summary so we can investigate and resolve it."
        ],
        "General": [
            "I can help with general logistics support. Please tell me what you need, such as office hours, support contact details, or delivery questions.",
            "Thanks for your message. I can help with general support questions about our service, contact options, and operating hours."
        ],
        "Weird": [
            "I cannot process impossible or fictional delivery requests, but I can absolutely help with real shipment, tracking, and delivery issues. Please share the actual order details.",
            "That request is not something we can fulfill, but I can still help with genuine logistics questions such as tracking, delivery timing, or parcel issues."
        ],
        "Typos": [
            "Thanks for the message. I understand you are asking about a package or shipment issue. Please share your tracking number or order details so I can help.",
            "I think you need support for a shipment question. Please send the order or tracking number, and I will review it for you."
        ],
        "Angry": [
            "I am very sorry for the inconvenience and understand your frustration. Please share your order number or tracking details so I can investigate and resolve the issue as quickly as possible.",
            "We are sorry for the delay and the frustration this has caused. Please share the order reference and I will review the case urgently."
        ],
        "default": [
            "I am here to help with logistics support. Please share your order number, tracking number, or a brief description of the issue.",
            "I did not catch that clearly. Please rephrase your question with your order details, tracking number, or parcel issue, and I will help."
        ],
        "tracking_request": [
            "I can help you track the parcel. Please share your tracking number, for example TRK1010, and I will check the latest status.",
            "Sure, I can look up the shipment status. Please send the tracking number so I can continue."
        ],
        "estimated_delivery": [
            "I can check the estimated delivery time. Please share your tracking number, for example TRK1010.",
            "I can look up when the parcel should arrive. Please send the tracking number so I can continue."
        ],
        "expedite_delivery": [
            "I can help speed up the delivery. Please share your tracking number, for example TRK1010, and I will submit a priority request.",
            "I can request faster delivery. Please send the tracking number so I can continue."
        ]
    }
    return responses


def normalize_dataset(df):
    """Normalize the CSV into a standard training format with intent and utterance columns."""
    column_map = {str(column).lower().strip(): column for column in df.columns}

    if "intent" in column_map:
        df = df.rename(columns={column_map["intent"]: "intent"})
    if "utterance" in column_map:
        df = df.rename(columns={column_map["utterance"]: "utterance"})
    if "question" in column_map:
        df = df.rename(columns={column_map["question"]: "utterance"})

    if "intent" not in df.columns:
        raise ValueError(f"The dataset does not contain an 'intent' column. Available columns: {list(df.columns)}")
    if "utterance" not in df.columns:
        raise ValueError(f"The dataset does not contain an 'utterance' or 'question' column. Available columns: {list(df.columns)}")

    df = df[["intent", "utterance"]].copy()
    df["intent"] = df["intent"].astype(str).str.strip()
    df["utterance"] = df["utterance"].astype(str).str.strip()
    intent_aliases = {
        "tracking_request": "Tracking",
        "tracking": "Tracking",
        "track order": "Tracking",
        "Track Order": "Tracking",
        "estimated_delivery": "Delivery",
        "estimated delivery": "Delivery",
        "ETA": "Delivery",
        "Delivery": "Delivery",
        "expedite delivery": "expedite_delivery",
        "expedite": "expedite_delivery",
    }
    df["intent"] = df["intent"].replace(intent_aliases)
    df = df.dropna(subset=["intent", "utterance"])
    df = df[df["intent"] != ""]
    df = df[df["utterance"] != ""]
    return df.reset_index(drop=True)


def train_logistics_model():
    dataset_path = find_dataset_path()
    print(f"--> Loading chatbot training dataset from: {dataset_path}")
    raw_df = pd.read_csv(dataset_path)
    df = normalize_dataset(raw_df)
    os.makedirs(NLP_DIR, exist_ok=True)

    quality = inspect_dataset_quality(df)
    print("--> Running dataset quality checks (no automatic deletion of useful rows)...")

    # Exact duplicate pairs add no new signal and can leak into both splits.
    split_df = df.drop_duplicates(subset=["intent", "utterance"], keep="first").reset_index(drop=True)
    quality["rows_after_exact_duplicate_drop_for_split"] = int(len(split_df))
    quality["exact_duplicates_removed_for_split_only"] = int(len(df) - len(split_df))

    X = split_df["utterance"]
    y = split_df["intent"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"--> Dataset split complete. Train size: {len(X_train)}, Test size: {len(X_test)}")

    leakage = overlap_leakage(X_train, X_test)
    quality["train_test_leakage"] = leakage
    write_json(QUALITY_PATH, quality)

    pd.DataFrame({"intent": y_train, "utterance": X_train}).to_csv(TRAIN_SPLIT_PATH, index=False)
    pd.DataFrame({"intent": y_test, "utterance": X_test}).to_csv(TEST_SPLIT_PATH, index=False)

    print("--> Vectorizing text using TF-IDF (shared preprocessor, ngram_range=(1, 2), max_features=8000)...")
    vectorizer = build_tfidf_vectorizer(ngram_range=(1, 2), max_features=8000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print("--> Training LinearSVC model for stronger intent separation on short support messages...")
    model = LinearSVC(class_weight="balanced", random_state=42)
    model.fit(X_train_tfidf, y_train)

    y_pred = model.predict(X_test_tfidf)
    labels = sorted(y.unique(), key=str)
    metrics = compute_holdout_metrics(y_test, y_pred, labels)
    metrics["train_size"] = int(len(X_train))
    metrics["test_size"] = int(len(X_test))
    metrics["total_samples"] = int(len(split_df))
    metrics["n_intents"] = int(y.nunique())
    metrics["split"] = "80/20 stratified, random_state=42"
    metrics["model"] = "TF-IDF + LinearSVC"
    metrics["tfidf"] = {
        "ngram_range": [1, 2],
        "max_features": 8000,
        "stop_words": "domain-safe list applied in the tokenizer (articles/copulas; WH-words kept)",
    }
    metrics["preprocessing"] = (
        "clean + lowercase + tokenize + English stop-word removal + WordNet lemmatization"
    )

    print("\n=================== MODEL PERFORMANCE REPORT ===================")
    print(classification_report(y_test, y_pred, zero_division=0))
    print(f"Overall Accuracy: {metrics['accuracy']:.4f}")
    print("================================================================\n")

    holdout_rows = []
    for utterance, actual, predicted in zip(X_test.tolist(), y_test.tolist(), y_pred.tolist()):
        transformed = vectorizer.transform([utterance])
        confidence, margin = linear_svc_decision_scores(model, transformed)
        holdout_rows.append(
            {
                "user_input": utterance,
                "expected_intent": actual,
                "predicted_intent": predicted,
                "correct": str(actual) == str(predicted),
                "decision_confidence": round(confidence, 4),
                "decision_margin": round(margin, 4),
                "preprocessed": preprocess_text(utterance),
            }
        )
    pd.DataFrame(holdout_rows).to_csv(HOLDOUT_PRED_PATH, index=False)

    _, png_path = save_confusion_matrix(y_test, y_pred, labels)
    metrics["confusion_matrix_csv"] = CONFUSION_CSV_PATH
    metrics["confusion_matrix_png"] = png_path

    # ---------- BASELINE COMPARISON: TF-IDF + Logistic Regression (item 14) ----------
    # LinearSVC stays as the main deployed model. LogReg is trained ONLY for the
    # assignment-required model-comparison table (per-group different solutions).
    print("\n--> Training baseline TF-IDF + Logistic Regression for model comparison only...")
    logreg = LogisticRegression(
        class_weight="balanced",
        random_state=42,
        max_iter=1000,
        solver="lbfgs",
    )
    logreg.fit(X_train_tfidf, y_train)
    y_pred_logreg = logreg.predict(X_test_tfidf)
    logreg_metrics = compute_holdout_metrics(y_test, y_pred_logreg, labels)
    print("--> LogReg holdout: accuracy {:.4f}  F1-weighted {:.4f}".format(
        logreg_metrics.get("accuracy", 0), logreg_metrics.get("f1_weighted", 0)
    ))
    print("--- TF-IDF + LinearSVC  vs  TF-IDF + Logistic Regression ---")
    for name in ["accuracy", "precision_weighted", "recall_weighted", "f1_weighted"]:
        print(
            "  {:<22}  LinearSVC = {:.4f}   LogReg = {:.4f}   Winner = {}".format(
                name + ":",
                float(metrics.get(name, 0) or 0),
                float(logreg_metrics.get(name, 0) or 0),
                "LinearSVC"
                if float(metrics.get(name, 0) or 0) >= float(logreg_metrics.get(name, 0) or 0)
                else "LogisticRegression",
            )
        )
    write_model_comparison(metrics, logreg_metrics, MODEL_COMPARISON_PATH, MODEL_COMPARISON_JSON)
    metrics["comparison_models"] = ["TF-IDF + LinearSVC", "TF-IDF + Logistic Regression"]
    metrics["comparison_winner_weighted_f1"] = (
        "TF-IDF + LinearSVC"
        if float(metrics.get("f1_weighted", 0) or 0) >= float(logreg_metrics.get("f1_weighted", 0) or 0)
        else "TF-IDF + Logistic Regression"
    )

    write_json(METRICS_PATH, metrics)

    print("--> Refitting on the full labelled dataset so short assignment phrases are remembered...")
    vectorizer = build_tfidf_vectorizer(ngram_range=(1, 2), max_features=8000)
    X_all_tfidf = vectorizer.fit_transform(X)
    model = LinearSVC(class_weight="balanced", random_state=42)
    model.fit(X_all_tfidf, y)

    os.makedirs("model", exist_ok=True)

    print("--> Saving response templates to data/intents.json...")
    response_data = {"responses": build_response_templates()}
    with open("data/intents.json", "w", encoding="utf-8") as f:
        json.dump(response_data, f, indent=2, ensure_ascii=False)

    print("--> Saving Vectorizer to model/vectorizer.pkl...")
    with open("model/vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    print("--> Saving Model to model/chatbot_model.pkl...")
    with open("model/chatbot_model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("--> Running paraphrase / typo / unclear robustness cases on the live chatbot...")
    from chatbot import LogisticsChatbot

    bot = LogisticsChatbot(confidence_threshold=0.30)
    robustness_rows = []
    relevance_triples = []
    unknown_fallback_substring = "I'm not quite sure how to help with that"
    for text, expected, case_type in ROBUSTNESS_CASES:
        bot.reset_session_context()
        reply, predicted, confidence = bot.get_bot_response(text)
        analysis = bot.last_nlp_analysis or {}
        predicted_name = str(predicted)
        expected_name = str(expected)
        correct = predicted_name == expected_name
        if expected in {"unknown", "unclear"}:
            correct = predicted_name in {"unknown_fallback", "Weird", "Typos", "General", "default"}
        is_unknown_fallback = (
            isinstance(reply, str)
            and unknown_fallback_substring.lower() in str(reply).lower()
        )
        relevance_triples.append((bool(correct), reply, is_unknown_fallback))
        robustness_rows.append(
            {
                "user_input": text,
                "expected_intent": expected,
                "predicted_intent": predicted,
                "correct": correct,
                "response": reply,
                "response_is_unknown_fallback": is_unknown_fallback,
                "decision_confidence": round(float(confidence or 0), 4),
                "fallback_used": bool(analysis.get("fallback_used")),
                "fallback_method": analysis.get("fallback_method"),
                "nlp_method": analysis.get("nlp_method"),
                "similarity_score": analysis.get("similarity_score"),
                "case_type": case_type,
                "preprocessed": analysis.get("preprocessed_text") or preprocess_text(text),
            }
        )
    pd.DataFrame(robustness_rows).to_csv(ROBUSTNESS_PATH, index=False)
    n_correct = sum(1 for r in robustness_rows if r["correct"])
    print("--> Robustness probes correct: {}/{}".format(n_correct, len(robustness_rows)))

    mapping_eval = evaluate_response_mapping(
        sorted(y.unique(), key=str), response_data["responses"], relevance_triples=relevance_triples
    )
    write_json(RESPONSE_EVAL_PATH, mapping_eval)

    print("[SUCCESS] Model training pipeline executed successfully!")


if __name__ == "__main__":
    train_logistics_model()
