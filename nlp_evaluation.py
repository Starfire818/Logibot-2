"""
NLP evaluation helpers for LogiBot.

Holdout metrics, robustness cases, dataset quality, and response-mapping checks.
BLEU/ROUGE are not used: replies are templates, CSV matches, and database lookups,
not generated against a reference translation.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


# ---------------------------------------------------------------------------
# Injected by build.py — packaged __file__-relative path resolution.
# ---------------------------------------------------------------------------
_EVAL_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
def _eval_resolve(rel_path):
    _abs = os.path.join(_EVAL_BASE_DIR, rel_path)
    if os.path.exists(_abs) or not os.path.exists(rel_path):
        return _abs
    return os.path.abspath(rel_path)

NLP_DIR = _eval_resolve("data/nlp")
METRICS_PATH = _eval_resolve(os.path.join("data/nlp", "metrics.json"))
QUALITY_PATH = _eval_resolve(os.path.join("data/nlp", "quality_report.json"))
HOLDOUT_PRED_PATH = _eval_resolve(os.path.join("data/nlp", "holdout_predictions.csv"))
ROBUSTNESS_PATH = _eval_resolve(os.path.join("data/nlp", "robustness_results.csv"))
CONFUSION_CSV_PATH = _eval_resolve(os.path.join("data/nlp", "confusion_matrix.csv"))
CONFUSION_PNG_PATH = _eval_resolve(os.path.join("data/nlp", "confusion_matrix.png"))
TRAIN_SPLIT_PATH = _eval_resolve(os.path.join("data/nlp", "train_split.csv"))
TEST_SPLIT_PATH = _eval_resolve(os.path.join("data/nlp", "test_split.csv"))
RESPONSE_EVAL_PATH = _eval_resolve(os.path.join("data/nlp", "response_evaluation.json"))
MODEL_COMPARISON_PATH = _eval_resolve(os.path.join("data/nlp", "model_comparison.csv"))
MODEL_COMPARISON_JSON = _eval_resolve(os.path.join("data/nlp", "model_comparison.json"))
USER_FEEDBACK_PATH = _eval_resolve(os.path.join("data", "user_feedback.csv"))


ROBUSTNESS_CASES = [
    # Same tracking intent, different wording (paraphrase group)
    ("Where is my parcel?", "tracking_request", "paraphrase"),
    ("Can you track my package?", "tracking_request", "paraphrase"),
    ("I want to know my shipment status.", "tracking_request", "paraphrase"),
    ("What's happening with my delivery?", "tracking_request", "paraphrase"),
    ("Track my parcel", "tracking_request", "short"),
    ("Please tell me the current location of my shipment and whether it is still moving.", "tracking_request", "long"),
    ("Wher is my pakage?", "tracking_request", "typo"),
    ("Where is my package at right now?", "tracking_request", "paraphrase"),
    ("Is my package still in transit?", "tracking_request", "paraphrase"),
    ("Can I get a status update on my order?", "tracking_request", "paraphrase"),
    # Estimated delivery paraphrases
    ("When will my parcel arrive?", "estimated_delivery", "paraphrase"),
    ("What is the estimated delivery time?", "estimated_delivery", "paraphrase"),
    ("When should I expect my package?", "estimated_delivery", "paraphrase"),
    ("eta for my order?", "estimated_delivery", "short"),
    ("What date will the package get to me?", "estimated_delivery", "paraphrase"),
    ("How many days until delivery?", "estimated_delivery", "paraphrase"),
    # Other logistics intents - paraphrase
    ("Can I change my address?", "Address Change", "paraphrase"),
    ("I need to update the shipping address please", "Address Change", "paraphrase"),
    ("My parcel is missing.", "Missing Parcel", "paraphrase"),
    ("I never received the package but it says delivered", "Missing Parcel", "paraphrase"),
    ("The box is crushed and the item is broken.", "Damaged Parcel", "paraphrase"),
    ("My order arrived damaged, what can I do?", "Damaged Parcel", "paraphrase"),
    ("How do I return this parcel?", "Returns", "paraphrase"),
    ("Can I get a refund for this order?", "Returns", "paraphrase"),
    ("Why is shipping so expensive?", "Shipping Cost", "paraphrase"),
    ("How much will it cost to ship to London?", "Shipping Cost", "paraphrase"),
    ("Can I reschedule pickup?", "Pickup", "paraphrase"),
    ("I missed the pickup today, what now?", "Pickup", "paraphrase"),
    ("Do you ship internationally?", "International", "paraphrase"),
    ("How long does shipping to Australia take?", "International", "paraphrase"),
    ("Why was my payment rejected?", "Payment", "paraphrase"),
    ("Can I pay with PayPal instead?", "Payment", "paraphrase"),
    ("I want to complain about the courier.", "Complaint", "paraphrase"),
    ("Your service has been absolutely terrible", "Complaint", "paraphrase"),
    ("What are your working hours?", "General", "paraphrase"),
    ("How can I contact customer support?", "General", "paraphrase"),
    ("Please speed up my delivery", "expedite_delivery", "paraphrase"),
    ("Can you make this delivery go faster?", "expedite_delivery", "paraphrase"),
    ("I need this parcel urgently, please hurry", "expedite_delivery", "paraphrase"),
    # Short questions
    ("Status?", "tracking_request", "short"),
    ("Track TRK1001", "tracking_request", "short"),
    ("ETA?", "estimated_delivery", "short"),
    ("Refund?", "Returns", "short"),
    # Long questions
    ("Hello I ordered a package last week on Tuesday and I was wondering if you could tell me where it is right now and if there have been any updates to the shipping status because I really need to receive it before the weekend starts thank you very much", "tracking_request", "long"),
    ("I would like to know approximately when my package should arrive at my home address because I need to make sure someone will be available to sign for the delivery and collect it from the courier", "estimated_delivery", "long"),
    # Typos and noisy input
    ("Wher is my pakage??", "tracking_request", "typo"),
    ("wen will mi parcel arriv?", "estimated_delivery", "typo"),
    ("trck my ordr plz", "tracking_request", "typo"),
    ("my paymnt was rejectd help", "Payment", "typo"),
    # Unclear / unknown / gibberish
    ("asdfgh", "unknown", "unclear"),
    ("Can you teleport my homework to Mars?", "Weird", "unknown"),
    ("hello??", "unknown", "unclear"),
    ("xyzzy plugh twist", "unknown", "unclear"),
    # Chinese queries (should be recognized via translation + keyword)
    ("我的包裹在哪里？", "tracking_request", "chinese"),
    ("我的包裹什么时候到？", "estimated_delivery", "chinese"),
]


def linear_svc_decision_scores(model, transformed):
    """Turn LinearSVC decision_function scores into a probability-like value and a margin."""
    scores = model.decision_function(transformed)
    scores = np.atleast_2d(np.asarray(scores, dtype=float))
    if scores.shape[0] == 0:
        return 0.0, 0.0

    row = scores[0]
    if row.size == 1:
        prob = float(1.0 / (1.0 + np.exp(-row[0])))
        return prob, abs(float(row[0]))

    shifted = row - np.max(row)
    exp_scores = np.exp(shifted)
    probs = exp_scores / np.sum(exp_scores)
    ordered = np.sort(row)
    margin = float(ordered[-1] - ordered[-2])
    return float(np.max(probs)), margin


def intent_correct(predicted, expected) -> bool:
    if expected in {"unknown", "unclear"}:
        return str(predicted).lower() in {
            "unknown_fallback",
            "weird",
            "typos",
            "general",
            "default",
            "empty_input",
        }
    left = str(predicted).strip().lower().replace("_", " ")
    right = str(expected).strip().lower().replace("_", " ")
    aliases = {
        "tracking": "tracking request",
        "track order": "tracking request",
        "eta": "estimated delivery",
        "delivery": "delivery",
    }
    left = aliases.get(left, left)
    right = aliases.get(right, right)
    return left == right or left.replace(" ", "") == right.replace(" ", "")


def compute_holdout_metrics(y_true, y_pred, labels):
    report = classification_report(
        y_true, y_pred, labels=labels, zero_division=0, output_dict=True
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "classification_report": report,
        "labels": list(labels),
    }


def save_confusion_matrix(y_true, y_pred, labels, csv_path=CONFUSION_CSV_PATH, png_path=CONFUSION_PNG_PATH):
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    frame = pd.DataFrame(matrix, index=labels, columns=labels)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    frame.to_csv(csv_path)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n_labels = max(len(labels), 1)
        fig_w = min(18, max(8, n_labels * 0.55))
        fig_h = min(16, max(7, n_labels * 0.5))
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        image = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks(range(n_labels))
        ax.set_yticks(range(n_labels))
        short_labels = [str(label).replace("_", " ") for label in labels]
        ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(short_labels, fontsize=8)
        ax.set_xlabel("Predicted Intent")
        ax.set_ylabel("Actual Intent")
        ax.set_title("Actual Intent vs Predicted Intent")

        thresh = matrix.max() / 2 if matrix.size else 0
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = int(matrix[i, j])
                if value == 0:
                    continue
                ax.text(
                    j,
                    i,
                    str(value),
                    ha="center",
                    va="center",
                    color="white" if value > thresh else "black",
                    fontsize=7,
                )
        fig.tight_layout()
        fig.savefig(png_path, dpi=140)
        plt.close(fig)
    except Exception:
        png_path = None

    return frame, png_path


def inspect_dataset_quality(df: pd.DataFrame) -> dict:
    """Report quality issues. Does not delete rows."""
    work = df.copy()
    work["intent"] = work["intent"].astype(str).str.strip()
    work["utterance"] = work["utterance"].astype(str)

    empty_text = int((work["utterance"].str.strip() == "").sum())
    missing_intent = int(work["intent"].isin(["", "nan", "None"]).sum())
    duplicate_rows = int(work.duplicated().sum())
    duplicate_utterances = int(work["utterance"].str.strip().str.lower().duplicated().sum())

    conflict = (
        work.assign(_u=work["utterance"].str.strip().str.lower())
        .groupby("_u")["intent"]
        .nunique()
    )
    conflicting_utterances = int((conflict > 1).sum())

    counts = work["intent"].value_counts()
    imbalance_ratio = float(counts.max() / max(counts.min(), 1)) if len(counts) else 0.0

    return {
        "total_rows": int(len(work)),
        "missing_utterances": empty_text,
        "missing_intent_labels": missing_intent,
        "empty_text": empty_text,
        "duplicate_rows": duplicate_rows,
        "duplicate_utterances": duplicate_utterances,
        "conflicting_duplicate_utterances": conflicting_utterances,
        "n_intents": int(work["intent"].nunique()),
        "samples_per_intent": {str(k): int(v) for k, v in counts.items()},
        "class_imbalance_ratio_max_over_min": round(imbalance_ratio, 3),
        "class_imbalance_note": (
            "LinearSVC is trained with class_weight='balanced'. Rows were not deleted."
            if imbalance_ratio >= 2
            else "Class counts are relatively even."
        ),
        "actions_taken": [
            "Exact duplicate (intent, utterance) pairs are dropped only for the train/test split to reduce leakage.",
            "Conflicting labels and rare classes are kept and reported, not auto-deleted.",
        ],
    }


def overlap_leakage(train_utterances, test_utterances) -> dict:
    train_set = {str(x).strip().lower() for x in train_utterances}
    test_set = {str(x).strip().lower() for x in test_utterances}
    overlap = sorted(train_set & test_set)
    return {
        "overlapping_utterances": len(overlap),
        "examples": overlap[:15],
        "note": (
            "Overlapping text in train and test inflates accuracy. Exact duplicate rows are dropped before splitting."
            if overlap
            else "No identical utterances appear in both the train and test splits."
        ),
    }


def evaluate_response_mapping(intent_names, responses: dict, relevance_triples=None) -> dict:
    """Check that classified intents can be mapped to a template. Not BLEU/ROUGE."""
    missing = []
    for intent in intent_names:
        keys = {
            intent,
            str(intent).replace("_", " ").title(),
            str(intent).replace(" ", "_"),
        }
        found = any(key in responses for key in keys)
        if not found:
            lower_map = {str(k).strip().lower().replace(" ", "_"): k for k in responses}
            found = str(intent).strip().lower().replace(" ", "_") in lower_map
        if not found:
            missing.append(intent)

    return {
        "method": "template_and_database_mapping",
        "bleu_rouge_used": False,
        "bleu_rouge_reason": (
            "LogiBot does not generate free text against a reference corpus. "
            "It selects predefined templates, similar dataset replies, or database fields. "
            "BLEU/ROUGE would not measure this system honestly."
        ),
        "intents_checked": len(list(intent_names)),
        "intents_missing_templates": missing,
        "template_coverage": (
            1.0 if not list(intent_names) else 1.0 - (len(missing) / max(len(list(intent_names)), 1))
        ),
        "relevance_evaluated": bool(relevance_triples),
        **(
            {}
            if not relevance_triples
            else {
                "relevance_total": len(relevance_triples),
                "relevance_relevant": sum(1 for t in relevance_triples if t and t[0] and not t[2]),
                "relevance_rate": (
                    0.0
                    if not relevance_triples
                    else sum(1 for t in relevance_triples if t and t[0] and not t[2]) / len(relevance_triples)
                ),
            }
        ),
    }


def write_model_comparison(linear_svc_metrics: dict, logreg_metrics: dict,
                           comparison_csv_path, comparison_json_path) -> pd.DataFrame:
    """Save a 2-row comparison table between LinearSVC and LogisticRegression baselines."""
    rows = [
        {
            "model": "TF-IDF + LinearSVC",
            "accuracy": linear_svc_metrics.get("accuracy"),
            "precision_macro": linear_svc_metrics.get("precision_macro"),
            "recall_macro": linear_svc_metrics.get("recall_macro"),
            "f1_macro": linear_svc_metrics.get("f1_macro"),
            "precision_weighted": linear_svc_metrics.get("precision_weighted"),
            "recall_weighted": linear_svc_metrics.get("recall_weighted"),
            "f1_weighted": linear_svc_metrics.get("f1_weighted"),
            "deployed_in_chatbot": True,
            "notes": (
                "Main classifier for LogiBot. Refit on full labelled dataset "
                "after holdout evaluation so short-support phrases are remembered."
            ),
        },
        {
            "model": "TF-IDF + Logistic Regression",
            "accuracy": logreg_metrics.get("accuracy"),
            "precision_macro": logreg_metrics.get("precision_macro"),
            "recall_macro": logreg_metrics.get("recall_macro"),
            "f1_macro": logreg_metrics.get("f1_macro"),
            "precision_weighted": logreg_metrics.get("precision_weighted"),
            "recall_weighted": logreg_metrics.get("recall_weighted"),
            "f1_weighted": logreg_metrics.get("f1_weighted"),
            "deployed_in_chatbot": False,
            "notes": (
                "Baseline comparison model only — trained on the same 80/20 split and the "
                "same shared TF-IDF vectorizer. NOT used for live chat responses (LogiBot "
                "keeps LinearSVC as its main intent classifier per project scope)."
            ),
        },
    ]
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(comparison_csv_path) or ".", exist_ok=True)
    df.to_csv(comparison_csv_path, index=False)
    write_json(comparison_json_path, {"comparison": rows})
    return df


def load_json(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, payload):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
