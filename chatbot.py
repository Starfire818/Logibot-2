"""
File: chatbot.py
Description: The core backend logical engine for parsing user inputs, 
             predicting intents, handling low-confidence fallbacks, and mapping 
             responses from JSON structure. (Enhanced with Regex DB Lookup)
"""

import os
import json
import random
import pickle
import re  # Imported for regex tracking ID validation

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

import nlp_preprocess  # noqa: F401 — required so the pickled TF-IDF tokenizer can unpickle
from nlp_evaluation import linear_svc_decision_scores
from nlp_preprocess import preprocess_text


def _resolve_path(rel_path: str) -> str:
    """Resolve a project-relative path to an absolute path.

    Used both by the source tree and the Streamlit Cloud deploy package.
    Always resolves against the directory that contains chatbot.py so that
    ``streamlit run app.py`` (cwd = project root) and direct ``python
    chatbot.py`` execution both locate data/model assets correctly.
    """
    base = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.join(base, rel_path)
    if os.path.exists(candidate) or os.path.isdir(os.path.dirname(candidate)):
        return candidate
    cwd_candidate = os.path.abspath(rel_path)
    if os.path.exists(cwd_candidate):
        return cwd_candidate
    return candidate


class LogisticsChatbot:
    def __init__(self, confidence_threshold=0.30):
        # LinearSVC softmax scores are used as a decision value, not true probabilities.
        self.threshold = confidence_threshold
        self.similarity_threshold = 0.22
        self.margin_threshold = 0.15
        self.last_nlp_analysis = self._empty_nlp_analysis()
        self.vectorizer_path = _resolve_path('model/vectorizer.pkl')
        self.model_path = _resolve_path('model/chatbot_model.pkl')
        self.intents_json_path = _resolve_path('data/intents.json')
        self.db_json_path = _resolve_path('data/logistics_db.json') # Registered path for the mock database
        self.orders_csv_path = _resolve_path('data/orders.csv') # Editable order registry for dynamic updates
        self.dataset_path = _resolve_path('Logistics_Customer_Support_Dataset.csv')
        self.reply_dataset_path = _resolve_path('data/replies.csv')

        # Core assets holders
        self.vectorizer = None
        self.model = None
        self.responses = {}
        self.mock_db = {} # Holder for package tracking database
        self.intent_examples = {}
        self.reply_examples = {}
        self.session_context = {
            "questions": [],
            "tracking_ids": [],
            "addresses": [],
            "recent_orders": [],
            "last_known_order": None,
            "tracking_id": None,
            "order_id": None,
            "last_intent": None,
            "previous_intent": None,
        }
        self._actionable_intents = {
            "Tracking",
            "Delivery",
            "expedite_delivery",
            "order_summary",
        }

        # Bootstrap initialization
        self.load_artifacts()

    @staticmethod
    def _empty_nlp_analysis():
        return {
            "detected_intent": None,
            "nlp_method": None,
            "decision_confidence": None,
            "fallback_used": False,
            "fallback_method": None,
            "similarity_score": None,
            "ml_intent": None,
            "ml_confidence": None,
            "decision_margin": None,
            "preprocessed_text": None,
        }

    def _set_nlp_analysis(self, **kwargs):
        analysis = self._empty_nlp_analysis()
        analysis.update(kwargs)
        if analysis.get("decision_confidence") is not None:
            analysis["decision_confidence"] = float(analysis["decision_confidence"])
        if analysis.get("similarity_score") is not None:
            analysis["similarity_score"] = float(analysis["similarity_score"])
        self.last_nlp_analysis = analysis
        return analysis

    def reset_session_context(self):
        """Clear remembered tracking IDs, order IDs, and previous intent."""
        self.session_context = {
            "questions": [],
            "tracking_ids": [],
            "addresses": [],
            "recent_orders": [],
            "last_known_order": None,
            "tracking_id": None,
            "order_id": None,
            "last_intent": None,
            "previous_intent": None,
        }

    @staticmethod
    def save_feedback(
        *,
        user_msg: str,
        bot_reply: str,
        detected_intent,
        rating_1_5,
        helpful_bool=None,
        comment_str=None,
        feedback_csv: str | None = None,
    ) -> bool:
        """Append one user-satisfaction row to the CSV file. Used in Streamlit.

        Parameters
        ----------
        rating_1_5 : int
            1..5 inclusive. Invalid values are clamped with a warning.
        helpful_bool : bool | None
            True = helpful, False = unhelpful, None = skipped.
        comment_str : str | None
            Optional free-text comment.

        Returns
        -------
        True on success, False on I/O failure (never raises so demos never crash).
        """
        import csv
        import datetime as _dt

        # Resolve default path (Streamlit Cloud-safe)
        if feedback_csv is None:
            feedback_csv = _resolve_path("data/user_feedback.csv")

        try:
            try:
                rating = int(rating_1_5)
            except (TypeError, ValueError):
                rating = 0
            if rating < 1:
                rating = 1
            elif rating > 5:
                rating = 5

            os.makedirs(os.path.dirname(feedback_csv) or ".", exist_ok=True)
            file_exists = os.path.isfile(feedback_csv)
            with open(feedback_csv, "a", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                if not file_exists:
                    writer.writerow([
                        "timestamp_utc",
                        "user_message",
                        "bot_reply",
                        "detected_intent",
                        "rating_1_5",
                        "helpful_bool",
                        "comment",
                    ])
                row_values = [
                    _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
                    str(user_msg)[:5000],
                    str(bot_reply)[:5000],
                    "" if detected_intent is None else str(detected_intent)[:120],
                    int(rating),
                    "" if helpful_bool is None else ("yes" if bool(helpful_bool) else "no"),
                    "" if comment_str in (None, "") else str(comment_str)[:5000],
                ]
                writer.writerow(row_values)
                # --- packaged-build: mirror row to configured cloud remotes ---
                try:
                    _ok_mirror, _report_mirror = _mirror_feedback_to_remotes(row_values)
                except Exception:  # noqa: BLE001
                    _ok_mirror, _report_mirror = False, "mirror skip"
            return True
        except Exception as exc:  # pragma: no cover - never crash UI on I/O errors
            return False

    def load_artifacts(self):
        # Exception Handling: Validate model and config files existence
        if not os.path.exists(self.vectorizer_path) or not os.path.exists(self.model_path):
            raise FileNotFoundError("Trained model files are missing. Please run train_model.py first.")

        if not os.path.exists(self.intents_json_path):
            raise FileNotFoundError(f"Configuration file missing at {self.intents_json_path}")

        # NOTE: logistics_db.json is OPTIONAL in packaged builds. orders.csv is the primary order registry.
        # JSON data (if present) is merged as secondary, non-authoritative, below.

        # Load binary pkl objects
        with open(self.vectorizer_path, 'rb') as f:
            self.vectorizer = pickle.load(f)
        with open(self.model_path, 'rb') as f:
            self.model = pickle.load(f)

        # Load intent configuration mapping
        with open(self.intents_json_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
            self.responses = config_data.get("responses", {})

        # Load simulated logistics database records.
        # CSV is preferred as the editable source of truth for live order updates.
        self.mock_db = {}
        self.available_columns = set()

        if os.path.exists(self.orders_csv_path):
            df = pd.read_csv(self.orders_csv_path)
            self.available_columns = set(df.columns.str.strip())
            if not df.empty:
                for _, row in df.iterrows():
                    tracking_id = str(row.get('tracking_id', row.get('TrackingID', ''))).strip().upper()
                    if not tracking_id:
                        continue
                    order_data = {}
                    order_data['status'] = str(row.get('status', row.get('Status', 'Unknown'))).strip()
                    order_data['destination'] = str(row.get('destination', row.get('Destination', 'Unknown'))).strip()
                    order_data['eta'] = self._eta_from_row(row)
                    order_data['customer_name'] = str(row.get('customer_name', row.get('CustomerName', 'Unknown'))).strip()
                    order_data['last_update'] = str(row.get('last_update', row.get('LastUpdate', 'Unknown'))).strip()
                    if 'receiver_name' in self.available_columns:
                        order_data['receiver_name'] = str(row.get('receiver_name', '')).strip()
                    if 'sender_name' in self.available_columns:
                        order_data['sender_name'] = str(row.get('sender_name', '')).strip()
                    if 'phone_number' in self.available_columns:
                        order_data['phone_number'] = str(row.get('phone_number', '')).strip()
                    if 'address' in self.available_columns:
                        order_data['address'] = str(row.get('address', '')).strip()
                    self.mock_db[tracking_id] = order_data

        if os.path.exists(self.db_json_path):
            with open(self.db_json_path, 'r', encoding='utf-8') as f:
                db_data = json.load(f)
            json_orders = db_data.get("mock_database", {}).get("orders", {})
            for tracking_id, order_info in json_orders.items():
                if tracking_id not in self.mock_db:
                    self.mock_db[tracking_id] = order_info

        # Load the training dataset to support NLP-based question matching
        if os.path.exists(self.dataset_path):
            df = pd.read_csv(self.dataset_path)
            if {'Intent', 'Question'}.issubset(set(df.columns)):
                for intent_name, group in df.groupby('Intent'):
                    self.intent_examples[intent_name.strip()] = [
                        str(question).strip()
                        for question in group['Question'].dropna().tolist()
                        if str(question).strip()
                    ]

        if os.path.exists(self.reply_dataset_path):
            reply_df = pd.read_csv(self.reply_dataset_path)
            if {'Intent', 'Question', 'Reply'}.issubset(set(reply_df.columns)):
                for intent_name, group in reply_df.groupby('Intent'):
                    records = []
                    for _, row in group.iterrows():
                        question = str(row.get('Question', '')).strip()
                        reply = str(row.get('Reply', '')).strip()
                        if question and reply:
                            records.append({"question": question, "reply": reply})
                    if records:
                        self.reply_examples[intent_name.strip()] = records

        if "Tracking" in self.intent_examples:
            self.intent_examples.setdefault("Tracking", self.intent_examples["Tracking"])
        if "Delivery" in self.intent_examples:
            self.intent_examples.setdefault("Delivery", self.intent_examples["Delivery"])
        if "Tracking" in self.reply_examples:
            self.reply_examples.setdefault("Tracking", self.reply_examples["Tracking"])
        if "Delivery" in self.reply_examples:
            self.reply_examples.setdefault("Delivery", self.reply_examples["Delivery"])

    def _normalize_intent_name(self, intent_name):
        """Normalize model labels so they match the response template keys."""
        if intent_name is None:
            return "default"

        normalized = str(intent_name).strip()
        normalized = normalized.replace("_", " ").replace("-", " ")
        normalized = re.sub(r"\s+", " ", normalized)

        aliases = {
            "track order": "Tracking",
            "track": "Tracking",
            "tracking": "Tracking",
            "tracking request": "Tracking",
            "expedite delivery": "expedite_delivery",
            "expedite": "expedite_delivery",
            "estimated delivery": "Delivery",
            "eta": "Delivery",
            "order summary": "order_summary",
            "order lookup": "order_summary",
            "delivery options": "Delivery",
            "delivery": "Delivery",
            "change shipping address": "Address Change",
            "address change": "Address Change",
            "check refund policy": "Returns",
            "refund": "Returns",
            "returns": "Returns",
            "contact human agent": "General",
            "general": "General",
            "create account": "General",
            "payment issue": "Payment",
            "payment": "Payment",
            "pickup": "Pickup",
            "business": "Business",
            "complaint": "Complaint",
            "angry": "Angry",
            "typos": "Typos",
            "weird": "Weird",
            "international": "International",
            "shipping cost": "Shipping Cost",
            "damaged parcel": "Damaged Parcel",
            "missing parcel": "Missing Parcel",
        }

        lower_key = normalized.lower()
        if lower_key in aliases:
            return aliases[lower_key]

        return normalized.title() if normalized and normalized[0].isalpha() else normalized

    def _safe_text_for_windows(self, text):
        """Keep replies natural. Never expose tool/database labels such as [TRACKING] or [ETA]."""
        return self._naturalize_reply(text)

    def _naturalize_reply(self, text):
        """Convert internal/tool-style output into a clean user-facing sentence."""
        if text is None:
            return ""

        safe_text = str(text)
        emoji_replacements = {
            '🔍': '',
            '📦': '',
            '📍': '',
            '🗺️': '',
            '📅': '',
            '❌': '',
            '🤖': '',
            '✅': '',
            '⚠️': '',
        }
        for symbol, replacement in emoji_replacements.items():
            safe_text = safe_text.replace(symbol, replacement)

        safe_text = re.sub(
            r'\[(TRACKING|PACKAGE|LOCATION|ROUTE|ETA|ERROR|BOT|OK|WARNING|RECEIVER|SENDER|STATUS|DB|ORDER|ESTIMATED)\]\s*',
            '',
            safe_text,
            flags=re.IGNORECASE,
        )
        safe_text = re.sub(r'\[[A-Z]{2,20}\]\s*', '', safe_text)
        safe_text = re.sub(r'\s{2,}', ' ', safe_text).strip()
        return safe_text

    def _contains_chinese(self, text):
        """Return True if text contains any CJK Unified Ideographs (basic Chinese range)."""
        try:
            return any('\u4e00' <= ch <= '\u9fff' for ch in str(text))
        except Exception:
            return False

    def _translate_chinese_to_english(self, text):
        """Translate Chinese to English.

        Attempts to use googletrans if installed. If not available, falls back to
        a small handcrafted mapping for common phrases. If translation fails,
        returns the original text.
        """
        if not self._contains_chinese(text):
            return text

        # Try to use googletrans (optional dependency)
        try:
            from googletrans import Translator
            translator = Translator()
            res = translator.translate(text, src='zh-cn', dest='en')
            if hasattr(res, 'text') and res.text:
                return res.text
        except Exception:
            # ignore and fall back to mapping
            pass

        mapping = {
            '我的包裹在哪里': 'where is my parcel',
            '我的包裹在哪里？': 'where is my parcel',
            '我想查快递状态': 'I want to check shipment status',
            '我的包裹什么时候到': 'when will my parcel arrive',
            '我的包裹什么时候到？': 'when will my parcel arrive',
            '我还没有收到包裹': 'I have not received my parcel',
            '我还沒有收到包裹': 'I have not received my parcel',
            '没收到': 'not received',
            '未收到': 'not received',
            '这些句子无法被chatbot理解': 'these sentences cannot be understood by the chatbot',
            '修复一下': 'fix it',
            '帮我修一下': 'please fix',
            '请帮我查一下包裹': 'please check my parcel',
            '包裹什么时候送到': 'when will the parcel be delivered',
            '包裹状态更新了吗': 'has the parcel status been updated',
            '什么时候能收到我的包裹': 'when can I receive my parcel',
            '我想知道快递现在在哪': 'I want to know where the parcel is now',
            '商家显示已签收但我没收到': "it says delivered but I didn't receive it",
            '几时到': 'when will it arrive',
            '几时到？': 'when will it arrive',
            '几时会到': 'when will it arrive',
            '几时会到？': 'when will it arrive',
            '什么时候到': 'when will it arrive',
            '什么时候到？': 'when will it arrive',
            '我的订单几时到': 'when will my order arrive',
            '我的订单几时到？': 'when will my order arrive',
            '我的订单什么时候到': 'when will my order arrive',
            '我的订单什么时候到？': 'when will my order arrive',
            '预计什么时候送到': 'what is the estimated delivery time',
            '预计什么时候送到？': 'what is the estimated delivery time',
            '预计送达时间': 'estimated delivery time',
            '预计送达时间？': 'estimated delivery time',
            '预计什么时候到': 'what is the estimated arrival time',
            '预计什么时候到？': 'what is the estimated arrival time',
            '预计几时到': 'what is the estimated arrival time',
            '预计几时到？': 'what is the estimated arrival time',
            '估计什么时候到': 'what is the estimated arrival time',
            '估计什么时候到？': 'what is the estimated arrival time',
            '几天能到': 'how many days to arrive',
            '几天能到？': 'how many days to arrive',
            '几号到': 'what date will it arrive',
            '几号到？': 'what date will it arrive',
            '几号送到': 'what date will it be delivered',
            '几号送到？': 'what date will it be delivered',
            '哪天到': 'what day will it arrive',
            '哪天到？': 'what day will it arrive',
        }

        # Normalize by removing spaces and punctuation common in user input
        key = re.sub(r"[^\u4e00-\u9fff]+", '', str(text))
        return mapping.get(key, text)

    def _eta_from_row(self, row):
        """Prefer ESTIMATED from the latest orders.csv, then other ETA columns."""
        for column in ('ESTIMATED', 'Estimated', 'estimate_time', 'eta', 'ETA'):
            if column in getattr(row, 'index', []):
                value = row.get(column)
                if value is not None and str(value).strip() and str(value).strip().lower() not in {'nan', 'none'}:
                    return str(value).strip()
        return 'Unknown'

    def _format_estimate_time(self, raw_eta, use_chinese=False):
        """Format estimate_time value from CSV into a human-readable string.
        Supports YYYY-MM-DD -> '28 August 2026' (EN) or '2026年8月28日' (CN).
        Passes through 'Completed' or other non-date values unchanged (with translation).
        """
        if not raw_eta or raw_eta.strip().lower() in ('unknown', '', 'nan', 'none'):
            return None

        eta = str(raw_eta).strip()

        if eta.lower() == 'completed':
            return '已送达' if use_chinese else 'Completed'

        try:
            from datetime import datetime
            parsed = None
            for fmt in ('%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M', '%Y-%m-%d'):
                try:
                    parsed = datetime.strptime(eta, fmt)
                    break
                except ValueError:
                    continue
            if parsed is None:
                return eta
            if use_chinese:
                has_time = (parsed.hour != 0 or parsed.minute != 0)
                if has_time or (':' in eta):
                    return f"{parsed.year}年{parsed.month}月{parsed.day}日 {parsed.hour:02d}:{parsed.minute:02d}"
                return f"{parsed.year}年{parsed.month}月{parsed.day}日"
            else:
                month_names = ['January', 'February', 'March', 'April', 'May', 'June',
                               'July', 'August', 'September', 'October', 'November', 'December']
                has_time = (parsed.hour != 0 or parsed.minute != 0)
                if has_time or (':' in eta):
                    return f"{parsed.day} {month_names[parsed.month - 1]} {parsed.year} {parsed.hour:02d}:{parsed.minute:02d}"
                return f"{parsed.day} {month_names[parsed.month - 1]} {parsed.year}"
        except Exception:
            return eta

    def _is_eta_query(self, user_message, cleaned_message):
        """Return True if the user message is primarily an ETA / estimated arrival query.

        IMPORTANT: Questions about 'what is happening with my delivery / parcel now'
        are STATUS (tracking) questions, not ETA questions. ETA requires explicit
        timing language such as when / time / date / soon / today / tomorrow.
        """
        if not cleaned_message:
            return False
        lowered = cleaned_message.lower().strip()
        cleaned_no_punct = re.sub(r'[?!.,:;？！。，：；]+$', '', lowered).strip()

        status_override_patterns = [
            r'\b(what|where)\b.*\b(happen|happening|going|status|location|position|move|moving|stuck|update)\b.*\b(delivery|parcel|package|order|shipment)\b',
            r'\b(delivery|parcel|package|order|shipment)\b.*\b(happen|happening|status|location|update|moving|stuck)\b',
            r'\bwhat\b.*\bwrong\b.*\b(delivery|parcel|package|order|shipment)\b',
            r'\b(status|location|where|track|tracking|update)\b.*\b(my|the|this)\b.*\b(parcel|package|order|shipment|delivery)\b',
            r'\b(can|may|could)\b.*\b(get|have|know|see|check)\b.*\b(status|update|location|where|tracking)\b',
            r'\bstatus\s+update\b',
            r'\bdamaged\b.*\b(parcel|package|order|item|box|shipment)\b',
            r'\b(parcel|package|order|item|box|shipment)\b.*\bdamaged\b',
        ]
        for pat in status_override_patterns:
            if re.search(pat, cleaned_no_punct, re.IGNORECASE):
                return False

        en_eta_patterns = [
            r'\bwhen\b.*\b(arrive|arrives|arrived|arrival|deliver|delivers|delivered|delivery|get|receive|receives|received|come|comes|came|eta|estimated|expect|expected|reach|reaches|reached)\b',
            r'\b(arrive|arrives|arrived|arrival|deliver|delivers|delivered|delivery|eta|estimated|expect|expected)\b.*\b(time|date|when|day)\b',
            r'\bwhat\b.*\b(estimated|eta|arrival|delivery|expected)\b',
            r'\bwhat\b.*\b(eta|arrival|delivery|deliver|arrive)\b.*\b(for|of|my|order|parcel|package|shipment)\b',
            r'\bhow\s+long\b.*\b(take|takes|took|arrive|arrives|arrived|deliver|delivers|delivered)\b',
            r'\b(days|date|day)\b.*\b(arrive|arrives|arrived|deliver|delivers|delivered|delivery)\b',
            r'^\s*eta\s*[\?\.]?\s*$',
            r'\beta\b.*\b(for|of|my|order|parcel|package|tracking|shipment)\b',
            r'\b(estimated|expected|expect)\b.*\b(delivery|arrival|date|time)\b',
            r'\b(soon|today|tomorrow|next week|this week|by date)\b.*\b(arrive|arrives|arrived|deliver|delivers|delivered|delivery)\b',
            r'\bwhat.*\btime\b.*\b(arrive|arrives|arrived|deliver|delivers|delivered|delivery)\b',
            r'\bwill\s+it\s+(arrive|arrives|arrived|deliver|delivers|delivered|come|comes|came|reach|reaches|reached|get|gets|got)\b',
            r'\b(can|could|should)\b.*\b(get|receive|expect)\b.*\b(my|the)\b.*\b(order|parcel|package|shipment|it)\b.*\b(by|before|at|on|this|next|today|tomorrow|time|date|soon)\b',
        ]
        for pat in en_eta_patterns:
            if re.search(pat, cleaned_no_punct, re.IGNORECASE):
                return True

        if self._contains_chinese(cleaned_message):
            cn_eta_keywords = [
                '几时', '什么时候', '何时', '预计', '估计', '几号', '哪天', '几日',
                '预计送达', '预计到达', '预计时间', '预计送到',
                '估计送到', '估计到达',
            ]
            msg_cn = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]+", '', cleaned_message)
            hit_count = sum(1 for kw in cn_eta_keywords if kw in msg_cn)
            if hit_count >= 1:
                if any(t in msg_cn for t in ['到', '送达', '送到', '到达', '时间', '日期', '时候']):
                    return True
                if hit_count >= 2:
                    return True
        return False

    def _extract_tracking_id_for_eta(self, user_message, cleaned_message):
        """Extract tracking ID from message, or fall back to session memory.
        Returns (tracking_id or None, source: 'message' or 'memory' or None).
        """
        m = re.search(r'\b(TRK\d{4})\b', cleaned_message, re.IGNORECASE)
        if m:
            return m.group(1).upper(), 'message'
        if self.session_context.get('tracking_id'):
            return self.session_context['tracking_id'], 'memory'
        if self.session_context.get('tracking_ids'):
            return self.session_context['tracking_ids'][-1], 'memory'
        if self.session_context.get('last_known_order'):
            return self.session_context['last_known_order'], 'memory'
        return None, None

    def _handle_eta_query(self, user_message, cleaned_message, force=False):
        """Main ETA query handler: find order -> read CSV estimate_time -> answer.
        Returns (reply_text or None, intent_name or None, confidence or None).
        If returns None, caller should fall through to normal ML pipeline.
        """
        if not force and not self._is_eta_query(user_message, cleaned_message):
            return None, None, None

        use_chinese = self._contains_chinese(cleaned_message)
        tracking_id, source = self._extract_tracking_id_for_eta(user_message, cleaned_message)

        if tracking_id is None:
            if use_chinese:
                reply = ("请问你的订单追踪号码是什么？请提供类似 TRK1001 这样的追踪号，我就可以帮你查询预计送达时间。"
                       " 你也可以先告诉我追踪号码，例如：「TRK1001 几时到？」")
            else:
                reply = ("What is your order tracking ID? Please share a tracking number like TRK1001 "
                       "and I can look up the estimated delivery time for you. You can also include it directly, "
                       "e.g. \"When will TRK1001 arrive?\"")
            return reply, 'Delivery', 0.95

        if tracking_id not in self.mock_db:
            if source == 'memory':
                memory_note_cn = f"（我记得你之前提到过 {tracking_id}，"
                memory_note_en = f"(I remember you mentioned {tracking_id} earlier. "
            else:
                memory_note_cn = f"（{tracking_id} "
                memory_note_en = f"({tracking_id} "
            if use_chinese:
                reply = (f"{memory_note_cn}但数据库中找不到这个订单的记录，暂时无法取得预计送达时间。)"
                       f" 请确认追踪号码是否正确。")
            else:
                reply = (f"{memory_note_en}but no records exist in our database, so I cannot retrieve "
                       f"the estimated delivery time right now. Please verify that the tracking number is correct.)")
            return reply, 'Delivery', 0.95

        order_info = self.mock_db[tracking_id]
        raw_eta = order_info.get('eta')
        formatted_eta = self._format_estimate_time(raw_eta, use_chinese=use_chinese)

        memory_prefix_cn = f"我记得你之前提到过 {tracking_id}。" if source == 'memory' else ""
        memory_prefix_en = f"I remember you mentioned {tracking_id} earlier. " if source == 'memory' else ""

        order_status = order_info.get('status', 'Unknown')
        destination = order_info.get('destination', 'Unknown')

        if formatted_eta is None or (isinstance(raw_eta, str) and raw_eta.strip().lower() in ('unknown', '', 'nan')):
            if use_chinese:
                reply = (f"{memory_prefix_cn}"
                       f"很抱歉，暂时无法取得订单 {tracking_id} 的预计送达时间。"
                       f" 这件包裹目前{order_status}，目的地是{destination}。")
            else:
                reply = (f"{memory_prefix_en}"
                       f"I could not find an estimated delivery date for {tracking_id} right now. "
                       f"The parcel is currently {order_status.lower()} and heading to {destination}.")
            return reply, 'Delivery', 0.98

        if use_chinese:
            if str(raw_eta).strip().lower() == 'completed':
                reply = f"{memory_prefix_cn}你的包裹 {tracking_id} 已经送达{destination}。"
            else:
                reply = (
                    f"{memory_prefix_cn}"
                    f"你的包裹 {tracking_id} 预计会在 {formatted_eta} 送达，"
                    f"目前{order_status}，目的地是{destination}。"
                )
        else:
            if str(raw_eta).strip().lower() == 'completed':
                reply = f"{memory_prefix_en}Parcel {tracking_id} has already been delivered to {destination}."
            else:
                reply = (
                    f"{memory_prefix_en}"
                    f"Parcel {tracking_id} is estimated to arrive on {formatted_eta}. "
                    f"It is currently {str(order_status).lower()} and heading to {destination}."
                )

        return reply, 'Delivery', 1.0

    def _is_tracking_status_query(self, user_message, cleaned_message):
        """Return True if message is asking about parcel location/status/tracking (not ETA)."""
        if not cleaned_message:
            return False
        lowered = cleaned_message.lower().strip()
        cleaned_no_punct = re.sub(r'[?!.,:;？！。，：；]+$', '', lowered).strip()

        en_status_patterns = [
            r'\b(where|location|position)\b.*\b(is|at|parcel|package|order|shipment)\b',
            # Noun side intentionally excludes "my" and "the" — otherwise innocent phrases
            # like "update the shipping address" match because update ∈ status verbs + "the".
            r'\b(track|tracking|status|update|check)\b.*\b(parcel|package|order|shipment)\b',
            r'\b(parcel|package|order|shipment)\b.*\b(status|location|where|track|update)\b',
            r'^\s*(where|track|tracking|status)\b',
            r'\btrack\s+(my\s+)?(parcel|package|order|shipment|delivery)\b',
            r'\b(in\s+transit|out\s+for\s+delivery|stuck|moving)\b',
            r'\b(what|which|current)\b.*\b(status|stage|step|progress)\b.*\b(parcel|package|order|shipment)\b',
        ]
        for pat in en_status_patterns:
            if re.search(pat, cleaned_no_punct, re.IGNORECASE):
                return True

        if self._contains_chinese(cleaned_message):
            cn_status_keywords_ind = ['在哪里', '在哪', '到哪了', '在哪儿', '哪了', '状态',
                                      '查快递', '查包裹', '查询', '物流', '快递', '包裹']
            msg_cn = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]+", '', cleaned_message)
            if any(kw in msg_cn for kw in cn_status_keywords_ind):
                return True
        return False

    def _build_natural_order_summary(self, order_info, tracking_id, focus=None, use_chinese=False):
        """Build a natural-language summary from order data, focused on user intent.
        focus: None = general summary, 'eta' = emphasize arrival date, 'status' = emphasize status/location.
        Never returns the raw [TRACKING]/[PACKAGE] labels.
        """
        status = str(order_info.get('status', 'Unknown')).strip()
        destination = str(order_info.get('destination', 'Unknown')).strip()
        raw_eta = order_info.get('eta')
        formatted_eta = self._format_estimate_time(raw_eta, use_chinese=use_chinese)
        status_lower = status.lower()

        if use_chinese:
            status_cn = status
            if status_lower == 'in transit':
                status_cn = '运输途中'
            elif status_lower == 'delivered':
                status_cn = '已送达'
            elif status_lower == 'processing in hub':
                status_cn = '在中转站处理中'
            elif status_lower == 'out for delivery':
                status_cn = '正在派送'
            elif status_lower == 'delayed':
                status_cn = '已延误'

            if focus == 'eta' and formatted_eta is not None:
                if str(raw_eta).strip().lower() == 'completed':
                    return f"你的包裹 {tracking_id} 已经送达{destination}。"
                return f"你的包裹 {tracking_id} 预计会在 {formatted_eta} 送达，目前正在{status_cn}，目的地是{destination}。"

            if str(raw_eta).strip().lower() == 'completed' or status_lower == 'delivered':
                return f"你的包裹 {tracking_id} 已经送达{destination}。"

            eta_part = f"，预计 {formatted_eta} 送达" if formatted_eta is not None else ""
            return f"你的包裹 {tracking_id} 目前{status_cn}，正在送往{destination}{eta_part}。"

        else:
            status_text = status.lower()
            if focus == 'eta' and formatted_eta is not None:
                if str(raw_eta).strip().lower() == 'completed':
                    return f"Parcel {tracking_id} has already been delivered to {destination}."
                return (
                    f"Parcel {tracking_id} is estimated to arrive on {formatted_eta}. "
                    f"It is currently {status_text} and heading to {destination}."
                )

            if str(raw_eta).strip().lower() == 'completed' or status_lower == 'delivered':
                return f"Parcel {tracking_id} has already been delivered to {destination}."

            eta_part = f" It is estimated to arrive on {formatted_eta}." if formatted_eta is not None else ""
            return f"Parcel {tracking_id} is currently {status_text} and heading to {destination}.{eta_part}"

    def _handle_tracking_only_query(self, user_message, cleaned_message, tracking_id, source='message'):
        """Handle the case where the message contains only/primarily a tracking ID
        (possibly with a simple status/ETA query). Builds a natural reply.
        Returns (reply_or_None, intent_or_None, conf_or_None)."""
        if tracking_id is None:
            return None, None, None

        if tracking_id not in self.mock_db:
            use_cn = self._contains_chinese(cleaned_message)
            if use_cn:
                if source == 'memory':
                    reply = f"我记得你之前提到过 {tracking_id}，但数据库中找不到这个订单的记录。请确认追踪号是否正确。"
                else:
                    reply = f"抱歉，我们的数据库中找不到订单 {tracking_id}。请确认追踪号码是否正确。"
            else:
                if source == 'memory':
                    reply = f"I remember you mentioned {tracking_id} earlier, but no records exist in our database for this order. Please verify the tracking number."
                else:
                    reply = f"Sorry, we couldn't find order {tracking_id} in our database. Please verify the tracking number."
            return reply, 'order_not_found', 1.0

        order_info = self.mock_db[tracking_id]
        use_cn = self._contains_chinese(cleaned_message)

        focus = None
        if self._is_eta_query(user_message, cleaned_message):
            focus = 'eta'
        elif self._is_tracking_status_query(user_message, cleaned_message):
            focus = 'status'

        reply = self._build_natural_order_summary(order_info, tracking_id, focus=focus, use_chinese=use_cn)
        if focus == 'eta':
            return reply, 'Delivery', 1.0
        if focus == 'status':
            return reply, 'Tracking', 1.0
        return reply, 'order_summary', 1.0

    def _get_response_for_intent(self, predicted_intent):
        """Fetch a response entry using a normalized intent key."""
        candidate_keys = [
            predicted_intent,
            self._normalize_intent_name(predicted_intent),
            str(predicted_intent).lower(),
            str(predicted_intent).strip().lower().replace(" ", "_"),
        ]

        normalized_mapping = {
            str(key).strip().lower().replace(" ", "_"): key
            for key in self.responses.keys()
        }

        for key in candidate_keys:
            key_str = str(key).strip().lower().replace(" ", "_")
            if key_str in normalized_mapping:
                actual_key = normalized_mapping[key_str]
                if actual_key in self.responses:
                    return random.choice(self.responses[actual_key])

        return random.choice(self.responses.get("default", ["I can help with your logistics question. Please share more details."]))

    def _build_contextual_reply(self, intent, user_message, matched_question=None):
        """Use NLP similarity and intent-specific phrases to create a more specific and human-like response."""
        base_reply = self._get_response_for_intent(intent)
        lowered_message = user_message.lower()

        keyword_map = {
            "Tracking": ["track", "where", "status", "parcel", "package", "moving", "stuck", "delivered", "包裹", "快递", "在哪里", "到哪", "在哪儿"],
            "Delivery": ["when", "arrive", "deliver", "late", "today", "tomorrow", "time", "什么时候", "何时 到达", "到达 时间"],
            "Address Change": ["address", "postcode", "office", "recipient", "phone", "apartment", "更改 地址", "改 地址", "收货 地址"],
            "Missing Parcel": ["missing", "never received", "delivered but", "stolen", "investigate", "没收到", "未收到", "找不到 包裹"],
            "Damaged Parcel": ["damaged", "broken", "crushed", "compensation", "claim", "破损", "损坏", "碎了"],
            "Returns": ["return", "refund", "cancel", "change my mind", "退货", "退款"],
            "Shipping Cost": ["shipping", "cost", "price", "discount", "free", "运费", "费用"],
            "Pickup": ["pickup", "collect", "reschedule", "missed", "自取", "上门 取件", "改 取件 时间"],
            "International": ["international", "customs", "tax", "documents", "import", "国际", "海关", "关税"],
            "Payment": ["pay", "card", "cash", "payment", "rejected", "支付", "付款", "扣款"],
            "Business": ["business", "bulk", "api", "warehouse", "daily", "企业", "批量 发货"],
            "Complaint": ["complaint", "terrible", "rude", "disappointed", "angry", "投诉", "差评", "不满意"],
            "General": ["hours", "open", "office", "support", "chat", "contact", "帮助", "支持"],
            "Weird": ["moon", "dinosaur", "invisible", "haunted", "underwater", "怪异"],
            "Typos": ["help", "late", "track", "please", "update", "修复", "fix it", "fix", "拼写 错误", "语法"],
            "Angry": ["unacceptable", "money back", "useless", "ridiculous", "waiting", "很生气", "退款", "太差"],
        }

        if intent in keyword_map and any(keyword in lowered_message for keyword in keyword_map[intent]):
            human_like_additions = {
                "Tracking": "I can check the latest movement of the parcel and explain what the current shipping status means.",
                "Delivery": "I can review the delivery ETA and confirm whether the parcel is still on schedule or delayed.",
                "Address Change": "I can check whether the parcel can still be redirected before it is handed to the courier.",
                "Missing Parcel": "I can help investigate whether the parcel was misdelivered, delayed, or lost during transit.",
                "Damaged Parcel": "I can help with the damage claim and guide you through the evidence steps for compensation.",
                "Returns": "I can help review the return window and refund timeline based on your order details.",
                "Shipping Cost": "I can explain the pricing factors and check whether any discount or promotion is available.",
                "Pickup": "I can help review the pickup schedule and whether the collection can be rebooked.",
                "International": "I can explain customs clearance, taxes, and the expected transit time for international shipments.",
                "Payment": "I can review the payment issue and explain the available payment options or next steps.",
                "Business": "I can help with bulk shipping, warehouse support, and API-related requirements for business accounts.",
                "Complaint": "I can escalate the case and make sure your complaint is reviewed properly by the right team.",
                "General": "I can answer general logistics questions and direct you to the correct support channel.",
                "Weird": "I can still assist with your real shipment issue, even if the request itself is unusual.",
                "Typos": "I can still understand the message and help with the parcel issue you are referring to.",
                "Angry": "I understand why you are upset, and I can help investigate this case urgently and professionally."
            }
            reply = base_reply + " " + human_like_additions.get(intent, "I can help with the next step for your case.")
            if matched_question:
                reply += f" Based on your message about '{matched_question}', I can focus on the most relevant next step for you."
            return reply

        if matched_question:
            return base_reply + f" Based on your message about '{matched_question}', I can help with the most relevant next step."

        return base_reply

    def _select_reply_from_dataset(self, intent, user_message):
        """Pick the most similar reply from the generated reply CSV using NLP similarity."""
        if intent not in self.reply_examples or not self.reply_examples[intent]:
            return None

        candidates = self.reply_examples[intent]
        questions = [item["question"] for item in candidates]

        try:
            candidate_matrix = self.vectorizer.transform(questions)
            input_vector = self.vectorizer.transform([user_message])
            similarities = cosine_similarity(input_vector, candidate_matrix)[0]
            best_index = int(np.argmax(similarities))
            if similarities[best_index] <= 0:
                return None
            return candidates[best_index]["reply"]
        except Exception:
            return None

    def _generate_nlp_reply(self, intent, user_message):
        """Use TF-IDF similarity to match the user message to similar dataset questions, then reply naturally."""
        dataset_reply = self._select_reply_from_dataset(intent, user_message)
        if dataset_reply:
            return dataset_reply

        if intent not in self.intent_examples or not self.intent_examples[intent]:
            return self._get_response_for_intent(intent)

        candidate_questions = self.intent_examples[intent]
        if not candidate_questions:
            return self._get_response_for_intent(intent)

        try:
            question_matrix = self.vectorizer.transform(candidate_questions)
            input_vector = self.vectorizer.transform([user_message])
            similarities = cosine_similarity(input_vector, question_matrix)[0]
            best_index = int(np.argmax(similarities))
            matched_question = candidate_questions[best_index]
            return self._build_contextual_reply(intent, user_message, matched_question)
        except Exception:
            return self._build_contextual_reply(intent, user_message)

    def _is_affirmation(self, text):
        """Return True for short confirmation replies such as yes / ok / 好的."""
        cleaned = re.sub(r'[?!.,:;？！。，：；]+', '', str(text).strip().lower())
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        affirmations = {
            'yes', 'y', 'yep', 'yeah', 'yup', 'ok', 'okay', 'sure', 'please',
            'confirm', 'confirmed', 'do it', 'go ahead', 'proceed', 'alright',
            'right', 'correct', 'of course', 'please do', 'yes please', 'yes ok',
            'ok please', 'sure please', 'go', 'continue',
            '好', '好的', '是', '是的', '可以', '嗯', '行', '对', '确认', '继续',
        }
        return cleaned in affirmations

    def _is_expedite_query(self, user_message, cleaned_message):
        """Return True if the user wants delivery sped up / prioritized."""
        if not cleaned_message:
            return False
        lowered = cleaned_message.lower()
        patterns = [
            r'\bspeed\s*up\b',
            r'\bexpedite\b',
            r'\brush\s+(deliver|delivery|ship|shipping|order|parcel|package)\b',
            r'\b(faster|quicker)\b.*\b(deliver|delivery|ship|shipping)\b',
            r'\b(deliver|delivery|ship|shipping)\b.*\b(faster|quicker|urgent(ly)?|asap|hurry|rush)\b',
            r'\burgent(ly)?\b.*\b(deliver|delivery|ship|shipping|parcel|package|order)\b',
            r'\b(parcel|package|order)\b.*\b(hurry|rush|urgent(ly)?)\b',
            r'\bexpress\s+(delivery|shipping)\b',
            r'\bpriority\s+(delivery|shipping)\b',
            r'\bmake\s+it\s+(faster|quicker|express)\b',
            r'\bneed\s+(this|my)\s+(parcel|package|order|delivery|shipment)\s+(urgent(ly)?|asap|hurry|rush)\b',
            r'\bplease\s+hurry\b',
        ]
        for pat in patterns:
            if re.search(pat, lowered, re.IGNORECASE):
                return True
        if self._contains_chinese(cleaned_message):
            cn_text = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]+", '', cleaned_message)
            if any(kw in cn_text for kw in ['加快', '加急', '催件', '催一下', '尽快送', '快点送', '加速配送']):
                return True
        return False

    def _is_bare_identifier_message(self, cleaned_message):
        """True when the message is mainly a tracking/order ID with little extra meaning."""
        if not cleaned_message:
            return False
        has_tracking = bool(re.search(r'\bTRK\d{4}\b', cleaned_message, re.IGNORECASE))
        has_order = bool(re.search(r'\b(?:ORD|ORDER)[-_ ]?\d{3,}\b', cleaned_message, re.IGNORECASE))
        if not has_tracking and not has_order:
            return False
        leftover = re.sub(r'\bTRK\d{4}\b', ' ', cleaned_message, flags=re.IGNORECASE)
        leftover = re.sub(r'\b(?:ORD|ORDER)[-_ ]?\d{3,}\b', ' ', leftover, flags=re.IGNORECASE)
        leftover = re.sub(r'[^\w\u4e00-\u9fff]+', ' ', leftover).strip().lower()
        allowed = {
            '', 'please', 'pls', 'check', 'this', 'order', 'parcel', 'package',
            'my', 'the', 'it', 'number', 'id', 'tracking',
        }
        tokens = leftover.split()
        return all(token in allowed for token in tokens)

    def _extract_ids_from_text(self, text):
        """Extract tracking_id and order_id from a user message."""
        tracking_id = None
        order_id = None
        tracking_match = re.search(r'\b(TRK\d{4})\b', str(text), re.IGNORECASE)
        if tracking_match:
            tracking_id = tracking_match.group(1).upper()
        order_match = re.search(r'\b((?:ORD|ORDER)[-_ ]?\d{3,})\b', str(text), re.IGNORECASE)
        if order_match:
            order_id = re.sub(r'[\s_-]+', '', order_match.group(1).upper()).replace('ORDER', 'ORD')
        return tracking_id, order_id

    def _resolve_tracking_id(self, cleaned_message):
        """Prefer an ID in the current message, then remembered session context."""
        tracking_id, _ = self._extract_ids_from_text(cleaned_message)
        if tracking_id:
            return tracking_id, 'message'
        if self.session_context.get('tracking_id'):
            return self.session_context['tracking_id'], 'memory'
        if self.session_context.get('last_known_order'):
            return self.session_context['last_known_order'], 'memory'
        if self.session_context.get('tracking_ids'):
            return self.session_context['tracking_ids'][-1], 'memory'
        return None, None

    def _remember_ids(self, tracking_id=None, order_id=None):
        if tracking_id:
            tracking_id = tracking_id.upper()
            self.session_context['tracking_id'] = tracking_id
            self.session_context['last_known_order'] = tracking_id
            if tracking_id not in self.session_context['tracking_ids']:
                self.session_context['tracking_ids'].append(tracking_id)
            if tracking_id not in self.session_context['recent_orders']:
                self.session_context['recent_orders'].append(tracking_id)
        if order_id:
            self.session_context['order_id'] = order_id

    def _set_last_intent(self, intent):
        if intent and intent not in {'affirmation', 'empty_input', 'unknown_fallback', 'conversation_context'}:
            self.session_context['previous_intent'] = self.session_context.get('last_intent')
            self.session_context['last_intent'] = intent

    def _finalize_response(self, reply, intent, confidence, tracking_id=None, order_id=None, nlp_analysis=None):
        self._remember_ids(tracking_id=tracking_id, order_id=order_id)
        self._set_last_intent(intent)
        payload = {
            "detected_intent": intent,
            "decision_confidence": confidence,
        }
        if nlp_analysis:
            payload.update(nlp_analysis)
            payload["detected_intent"] = intent
            if payload.get("decision_confidence") is None:
                payload["decision_confidence"] = confidence
        self._set_nlp_analysis(**payload)
        return self._naturalize_reply(reply), intent, confidence

    def _reload_orders_csv(self):
        """Reload the latest orders.csv so ETA/status answers stay current."""
        if not os.path.exists(self.orders_csv_path):
            return
        try:
            df = pd.read_csv(self.orders_csv_path)
            self.available_columns = set(df.columns.str.strip())
            if df.empty:
                return
            for _, row in df.iterrows():
                tracking_id = str(row.get('tracking_id', row.get('TrackingID', ''))).strip().upper()
                if not tracking_id:
                    continue
                order_data = {
                    'status': str(row.get('status', row.get('Status', 'Unknown'))).strip(),
                    'destination': str(row.get('destination', row.get('Destination', 'Unknown'))).strip(),
                    'eta': self._eta_from_row(row),
                    'customer_name': str(row.get('customer_name', row.get('CustomerName', 'Unknown'))).strip(),
                    'last_update': str(row.get('last_update', row.get('LastUpdate', 'Unknown'))).strip(),
                }
                if 'receiver_name' in self.available_columns:
                    order_data['receiver_name'] = str(row.get('receiver_name', '')).strip()
                if 'sender_name' in self.available_columns:
                    order_data['sender_name'] = str(row.get('sender_name', '')).strip()
                if 'phone_number' in self.available_columns:
                    order_data['phone_number'] = str(row.get('phone_number', '')).strip()
                if 'address' in self.available_columns:
                    order_data['address'] = str(row.get('address', '')).strip()
                self.mock_db[tracking_id] = order_data
        except Exception:
            pass

    def _ask_for_tracking_id(self, intent, use_chinese=False):
        if use_chinese:
            if intent == 'expedite_delivery':
                return "好的，我可以帮你加快配送。请提供追踪号码，例如 TRK1010。"
            if intent == 'Delivery':
                return "请问你的订单追踪号码是什么？请提供类似 TRK1010 的追踪号，我就可以查询预计送达时间。"
            return "好的，我可以帮你查询包裹。请提供追踪号码，例如 TRK1010。"
        if intent == 'expedite_delivery':
            return "I can help speed up the delivery. Please share the tracking number, for example TRK1010."
        if intent == 'Delivery':
            return "I can check the estimated delivery time. Please share the tracking number, for example TRK1010."
        return "I can help you track the parcel. Please share the tracking number, for example TRK1010."

    def _handle_expedite_delivery(self, user_message, cleaned_message, tracking_id, source='message'):
        use_chinese = self._contains_chinese(cleaned_message)
        if tracking_id is None:
            return self._ask_for_tracking_id('expedite_delivery', use_chinese), 'expedite_delivery', 0.96

        if tracking_id not in self.mock_db:
            if use_chinese:
                reply = f"抱歉，数据库中找不到订单 {tracking_id}，暂时无法加快配送。请确认追踪号码是否正确。"
            else:
                reply = f"I could not find order {tracking_id}, so I cannot speed up that delivery yet. Please check the tracking number and try again."
            return reply, 'expedite_delivery', 0.96

        order_info = self.mock_db[tracking_id]
        status = str(order_info.get('status', 'Unknown')).strip()
        destination = str(order_info.get('destination', 'Unknown')).strip()
        formatted_eta = self._format_estimate_time(order_info.get('eta'), use_chinese=use_chinese)
        memory_prefix = ''
        if source == 'memory':
            memory_prefix = f"我记得你之前提到过 {tracking_id}。" if use_chinese else f"I'll continue with {tracking_id} from earlier. "

        if status.lower() == 'delivered' or str(order_info.get('eta', '')).strip().lower() == 'completed':
            if use_chinese:
                reply = f"{memory_prefix}订单 {tracking_id} 已经送达{destination}，所以不需要再加快配送。"
            else:
                reply = f"{memory_prefix}Parcel {tracking_id} has already been delivered to {destination}, so it cannot be expedited."
            return reply, 'expedite_delivery', 1.0

        eta_text = ''
        if formatted_eta:
            eta_text = f" 目前预计 {formatted_eta} 送达。" if use_chinese else f" It is currently estimated to arrive on {formatted_eta}."

        if use_chinese:
            reply = (
                f"{memory_prefix}我已经为订单 {tracking_id} 提交了加快配送请求。"
                f" 当前状态是{status}，目的地是{destination}。{eta_text}"
                " 我们会尽量优先处理这件包裹。"
            )
        else:
            reply = (
                f"{memory_prefix}I've submitted a request to speed up delivery for {tracking_id}. "
                f"It is currently {status} and heading to {destination}.{eta_text} "
                "Our team will try to prioritize this shipment."
            )
        return reply, 'expedite_delivery', 1.0

    def _meaning_key(self, text):
        """Normalize a message to its request meaning, ignoring tracking/order IDs."""
        stripped = re.sub(r'\bTRK\d{4}\b', ' ', str(text), flags=re.IGNORECASE)
        stripped = re.sub(r'\b(?:ORD|ORDER)[-_ ]?\d{3,}\b', ' ', stripped, flags=re.IGNORECASE)
        stripped = re.sub(r'[?!.,:;\'\"？！。，：；]+', '', stripped)
        return re.sub(r'\s+', ' ', stripped).strip().lower()

    def _resolve_continued_intent(self):
        """Continue the previous logistics request using saved intent and prior questions."""
        for candidate in (
            self.session_context.get('last_intent'),
            self.session_context.get('previous_intent'),
        ):
            normalized = self._normalize_intent_name(candidate)
            if normalized in self._actionable_intents:
                return normalized

        questions = self.session_context.get('questions') or []
        for question in reversed(questions):
            if self._is_affirmation(question):
                continue
            semantic = self._classify_semantic_intent(question, question.strip())
            if semantic in self._actionable_intents:
                return semantic
            keyword_intent, _ = self._keyword_fallback_intent(question)
            keyword_intent = self._normalize_intent_name(keyword_intent)
            if keyword_intent in self._actionable_intents:
                return keyword_intent
        return None

    def _classify_semantic_intent(self, user_message, cleaned_message):
        """Classify from the meaning of the whole sentence, not from ID presence alone.

        Order of checks (intent-priority order, most specific first):
          1. Affirmation
          2. Exact phrase lookup
          3. Address Change (explicit "update shipping address" etc.) — before tracking-status
             because "shipping address" substring would otherwise be caught by tracking.
          4. International shipping with explicit destination country — before ETA, because
             "how long does shipping to Australia take?" is about International, not plain ETA.
          5. Expedite / express delivery
          6. ETA / arrival-time query
          7. Tracking / status / what's-happening-with-delivery query
          8. Bare TRACK word
          9. Bare identifier message
        """
        if self._is_affirmation(cleaned_message):
            return 'affirmation'

        meaning = self._meaning_key(cleaned_message)
        phrase_intents = {
            'track my parcel': 'Tracking',
            'track order': 'Tracking',
            'track my order': 'Tracking',
            'track my package': 'Tracking',
            'please track my parcel': 'Tracking',
            'i want to track my parcel': 'Tracking',
            'track parcel': 'Tracking',
            'speed up delivery': 'expedite_delivery',
            'speed up my delivery': 'expedite_delivery',
            'please speed up delivery': 'expedite_delivery',
            'when will it arrive': 'Delivery',
            'when will it be delivered': 'Delivery',
            'when will it be deliver': 'Delivery',
        }
        if meaning in phrase_intents:
            return phrase_intents[meaning]

        clean_msg = re.sub(r'[?!.,:;？！。，：；]+$', '', cleaned_message.lower()).strip()

        # --- 3. Address Change (explicit compound phrases) BEFORE tracking/ETA ---
        if re.search(r'\b(change|update|wrong|incorrect|new|fix)\b.*\b(address|adress|postcode|postal|recipient|apartment|street|city)\b', clean_msg) or \
           re.search(r'\b(address|adress)\b.*\b(change|update|wrong|incorrect|new|fix)\b', clean_msg) or \
           re.search(r'\b(shipping|delivery|ship|billing)\b.*\b(address|adress)\b', clean_msg):
            return 'Address Change'

        # --- 4. International shipping: destination country explicitly present BEFORE ETA ---
        has_international_long = any(kw in clean_msg for kw in [
            "international", "overseas", "customs", "import tax", "documents needed",
            "australia", "canada", "germany", "france", "japan", "china", "india",
            "united kingdom", "united states", "europe", "asia",
            "abroad", "foreign", "global", "worldwide",
        ])
        has_short_code = re.search(r'\b(uk|usa)\b', clean_msg) or re.search(r'\bus\b', clean_msg)
        has_destination_country = re.search(
            r'\b(ship|shipping|deliver|delivery|send|parcel|package|order|sending|shipping to)\b.*\b(to|from|for)\b.*\b(australia|canada|germany|france|japan|china|india|uk|usa|europe|asia|overseas|abroad)\b',
            clean_msg
        )
        if (has_international_long or has_short_code) and (
            re.search(r'\b(ship|shipping|deliver|delivery|send|cost|price|how long|how much|customs|tax|import)\b', clean_msg)
            or has_destination_country
        ):
            return 'International'

        if self._is_expedite_query(user_message, cleaned_message):
            return 'expedite_delivery'
        if self._is_eta_query(user_message, cleaned_message):
            return 'Delivery'
        # Explicit "what's happening / what's going on with delivery" -> tracking (not ETA, not expedite)
        if re.search(r'\bwhat\b.*\b(happen|happening|going on|status|update|wrong)\b.*\b(delivery|parcel|package|order|shipment)\b', clean_msg):
            return 'Tracking'
        if self._is_tracking_status_query(user_message, cleaned_message):
            return 'Tracking'
        if re.search(r'^\s*track(\s+(my\s+)?(parcel|package|order|shipment|delivery))?\s*$', cleaned_message, re.I):
            return 'Tracking'
        if self._is_bare_identifier_message(cleaned_message):
            return 'order_summary'
        return None

    def _dispatch_logistics_intent(self, intent, user_message, cleaned_message, tracking_id, source):
        use_chinese = self._contains_chinese(cleaned_message)
        if intent == 'expedite_delivery':
            return self._handle_expedite_delivery(user_message, cleaned_message, tracking_id, source=source or 'message')
        if intent == 'Delivery':
            reply, eta_intent, conf = self._handle_eta_query(user_message, cleaned_message, force=True)
            return reply, 'Delivery', conf or 0.96
        if intent == 'Tracking':
            if tracking_id is None:
                return self._ask_for_tracking_id('Tracking', use_chinese), 'Tracking', 0.96
            reply, _, conf = self._handle_tracking_only_query(
                user_message, cleaned_message, tracking_id, source=source or 'message'
            )
            if source == 'memory' and reply:
                prefix = f"我记得你之前提到过 {tracking_id}。" if use_chinese else f"I'll continue with {tracking_id} from earlier. "
                reply = prefix + reply
            return reply, 'Tracking', conf or 0.96
        if intent == 'order_summary':
            if tracking_id is None:
                return self._ask_for_tracking_id('Tracking', use_chinese), 'order_summary', 0.9
            return self._handle_tracking_only_query(
                user_message, cleaned_message, tracking_id, source=source or 'message'
            )
        return None, None, None

    def _keyword_fallback_intent(self, user_message):
        """Use keyword-based intent matching before the ML model fallback to catch short or noisy messages.

        Priority: ETA/Delivery > Tracking/Status > Missing Parcel requires EXPLICIT missing
        wording (not just 'when will it arrive'). This avoids classifying ETA questions as Missing Parcel.
        """
        lower_message = user_message.lower()
        cleaned_strip = user_message.strip()
        clean_msg = re.sub(r'[?!.,:;？！。，：；]+$', '', lower_message).strip()

        # ---------- PRIORITY 0: Expedite delivery ----------
        if self._is_expedite_query(user_message, cleaned_strip):
            return "expedite_delivery", 0.24

        # ---------- PRIORITY 0.3: Explicit damage report MUST come before international (short codes) ----------
        if re.search(r'\b(damaged|damagd|broken|broke|crushed|crushd|破损|损坏|compensation|claim)\b', clean_msg):
            return "Damaged Parcel", 0.20

        # ---------- PRIORITY 0.5: International shipping context overrides generic ETA when country/overseas present ----------
        # IMPORTANT: use word boundaries for short codes to avoid matching inside words like "customer"→"us", "broken"→"uk"
        has_international_long = any(kw in clean_msg for kw in [
            "international", "overseas", "customs", "import tax", "documents needed",
            "australia", "canada", "germany", "france", "japan", "china", "india",
            "united kingdom", "united states", "europe", "asia",
            "abroad", "foreign", "global", "worldwide",
            "国际", "海关", "关税", "海外",
        ])
        has_short_code = re.search(r'\b(uk|usa)\b', clean_msg) or \
                          re.search(r'\bus\b', clean_msg)  # standalone "us", not inside "customer"
        has_destination_country = re.search(
            r'\b(ship|shipping|deliver|delivery|send|parcel|package|order)\b.*\b(to|from|for)\b.*\b(australia|canada|germany|france|japan|china|india|uk|usa|europe|asia|overseas|abroad)\b',
            clean_msg
        )
        if has_international_long or has_short_code or has_destination_country:
            return "International", 0.21

        # ---------- PRIORITY 1: ETA / Arrival query -> estimated_delivery ----------
        if self._is_eta_query(user_message, cleaned_strip):
            return "Delivery", 0.22

        if re.search(r'\b(when|will|what\s+time|when\s+will|by\s+when)\b', clean_msg) and \
           re.search(r'\b(arrive|arrives|arrived|deliver|delivers|delivered|get|receive|receives|received|delivery|come|comes|came|reach|reaches|reached)\b', clean_msg):
            return "Delivery", 0.20

        # ---------- PRIORITY 2: Tracking / Status / Location queries ----------
        if re.search(r'\b(where|location|position)\b', clean_msg) and \
           re.search(r'\b(parcel|package|order|shipment|it)\b', clean_msg):
            return "Tracking", 0.20

        if re.search(r'\b(track|tracking)\b', clean_msg):
            return "Tracking", 0.22

        # IMPORTANT: Noun part must be parcel/package/order/shipment, NOT "my" or "the" —
        # otherwise "update the shipping address" hits tracking_request because update∈{status,update,check}
        # and "the" matches the loose noun part.
        if re.search(r'\b(track|tracking|status|update|check)\b.*\b(parcel|package|order|shipment)\b', clean_msg):
            return "Tracking", 0.20

        if self._is_tracking_status_query(user_message, cleaned_strip):
            return "Tracking", 0.20

        # ---------- PRIORITY 3: Missing Parcel ONLY when user explicitly reports missing/lost ----------
        missing_explicit = [
            "missing parcel", "missing package", "missing order", "missing shipment",
            "my parcel is missing", "my package is missing", "the parcel is missing",
            "never received", "not received", "hasn't arrived", "hasn't come", "haven't arrived",
            "haven't received", "did not receive", "didn't receive",
            "my parcel is lost", "my package is lost", "lost parcel", "lost package", "lost order",
            "stolen", "stole my parcel", "stole my package", "where is my missing",
            "investigate", "investigation",
            "delivered but i didn't", "delivered but not", "delivered but i haven't", "delivered but haven't",
            "没收到", "未收到", "找不到包裹", "包裹丢了", "丢了", "被盗", "一直 没到", "没 收到"
        ]
        if any(kw in lower_message for kw in missing_explicit):
            return "Missing Parcel", 0.20

        # ---------- PRIORITY 3.5: Explicit damage / address / returns / payment / pickup / international / complaint
        if re.search(r'\b(damaged|damagd|broken|broke|crushed|crushd|破损|损坏)\b', clean_msg) or \
           re.search(r'\b(compensation|compensate|claim|refund claim)\b', clean_msg):
            return "Damaged Parcel", 0.20
        if re.search(r'\b(change|update|wrong|incorrect|new)\b.*\b(address|adress|postcode|postal|recipient|apartment)\b', clean_msg) or \
           re.search(r'\b(address|adress)\b.*\b(change|update|wrong|incorrect|new)\b', clean_msg) or \
           re.search(r'\b(shipping|delivery|ship)\b.*\b(address|adress)\b', clean_msg) or \
           any(kw in lower_message for kw in ["改 地址", "更新 地址", "收货 地址", "地址 错了"]):
            return "Address Change", 0.20
        if re.search(r'\b(return|rturn|refund|rfund|cancel|exchange|退货|退款)\b', clean_msg):
            return "Returns", 0.20
        if re.search(r'\b(shipping|ship|delivery)\b.*\b(cost|price|fee|charge|discount|free|expensive|cheap)\b', clean_msg) or \
           re.search(r'\b(cost|price|fee|charge|discount|free)\b.*\b(shipping|ship|delivery)\b', clean_msg) or \
           any(kw in lower_message for kw in ["运费", "费用", "多少钱"]):
            return "Shipping Cost", 0.20
        if re.search(r'\b(pickup|pick up|pckup|collect|reschedule|missed pickup|自取|上门 取件)\b', clean_msg):
            return "Pickup", 0.20
        if re.search(r'\b(international|overseas|customs|import|export|tax|document|国际|海关|关税)\b', clean_msg):
            return "International", 0.20
        if re.search(r'\b(payment|pay|card|cash|reject|decline|paypal|支付|付款|扣款)\b', clean_msg) and \
           not re.search(r'\b(pay|payment)\b.*\b(cash on delivery)\b', clean_msg):
            return "Payment", 0.20
        if re.search(r'\b(complain|complaint|terrible|disappoint|angry|rude|unacceptable|useless|投诉|不满意|差劲)\b', clean_msg):
            return "Complaint", 0.20
        if re.search(r'\b(business|bulk|api|warehouse|daily|enterprise|企业|批量)\b', clean_msg):
            return "Business", 0.20
        if re.search(r'\b(urgent|urgently|hurry|faster|asap|rush|priority|express)\b.*\b(delivery|deliver|parcel|package|order|ship)\b', clean_msg) or \
           re.search(r'\b(delivery|deliver|parcel|package|order|ship)\b.*\b(urgent|urgently|hurry|faster|asap|rush|priority|express)\b', clean_msg):
            return "expedite_delivery", 0.22

        # ---------- PRIORITY 4: General keyword fallback (weaker confidence) ----------
        intent_keywords = {
            "Tracking": (["where is my parcel", "where is my package",
                                "package status", "shipment status", "in transit", "out for delivery",
                                "status update", "track my parcel", "track order", "track my order",
                                "包裹", "快递", "在哪里", "在哪儿"], 0.18),
            "Delivery":       (["when will", "arrive", "arrival", "arrived", "delivery time",
                                "deliver tomorrow", "late delivery", "today delivery", "weekend delivery",
                                "什么时候 到", "何时 到达", "几号 到", "预计", "几时"], 0.15),
            "Address Change": (["change address", "wrong address", "new address", "postcode",
                                "office instead", "recipient", "phone number", "apartment number",
                                "改 地址", "更新 地址"], 0.15),
            "Damaged Parcel": (["damaged", "broken", "crushed", "box is crushed",
                                "item inside is broken", "compensation", "claim",
                                "破损", "损坏"], 0.15),
            "Returns":        (["return", "refund", "cancel shipment", "cancel my shipment",
                                "change my mind", "return shipping", "退货", "退款"], 0.15),
            "Shipping Cost":  (["shipping", "cost", "price", "discount", "free",
                                "运费", "费用", "多少钱"], 0.15),
            "Pickup":         (["pickup", "pick up", "reschedule pickup", "missed pickup",
                                "leave my parcel", "自取", "上门 取件"], 0.15),
            "International":  (["international", "customs", "import tax", "documents needed",
                                "shipping overseas", "国际", "海关", "关税"], 0.15),
            "Payment":        (["pay cash", "cash on delivery", "credit card", "card",
                                "payment rejected", "pay later", "支付", "付款"], 0.15),
            "Business":       (["business account", "bulk shipping", "api integration",
                                "warehouse", "daily pickup", "企业", "批量 发货"], 0.15),
            "Complaint":      (["complaint", "terrible", "disappointed", "rude courier",
                                "nobody answered", "waited all day", "投诉", "不满意", "差劲"], 0.15),
            "General":        (["office", "support", "working hours", "open today",
                                "contact support", "live chat", "帮助", "支持"], 0.15),
            "Weird":          (["moon", "dinosaur", "invisible", "haunted", "underwater", "怪异"], 0.15),
            "Typos":          (["late again", "no update", "help", "fix it", "asap",
                                "修复", "帮我 修复", "拼写 错误"], 0.15),
            "Angry":          (["unacceptable", "money back", "useless", "ridiculous",
                                "waiting for days", "never using your service again",
                                "很生气", "退款", "太 差"], 0.15),
        }

        for intent, (keywords, score) in intent_keywords.items():
            if any(keyword in lower_message for keyword in keywords):
                return intent, score

        return None, 0.0

    def _fallback_similarity_intent(self, user_message):
        """Use per-intent TF-IDF similarity as a second-pass fallback when model confidence is low."""
        if not self.intent_examples:
            return None, 0.0

        try:
            best_intent = None
            best_score = 0.0
            input_vector = self.vectorizer.transform([user_message])

            for intent_name, questions in self.intent_examples.items():
                if not questions:
                    continue

                matrix = self.vectorizer.transform(questions)
                similarities = cosine_similarity(input_vector, matrix)[0]
                intent_score = float(np.max(similarities))

                if intent_score > best_score:
                    best_score = intent_score
                    best_intent = intent_name

            if best_intent is not None and best_score >= self.similarity_threshold:
                resolved_intent = self._disambiguate_similarity_result(
                    best_intent, best_score, user_message
                )
                return resolved_intent, best_score
        except Exception:
            pass

        return None, 0.0

    def _disambiguate_similarity_result(self, raw_intent, raw_score, user_message):
        """If cosine fallback lands on Typos or Weird, check keyword-based intent and prefer actionable logistics intents."""
        lower_name = str(raw_intent).strip().lower()
        if lower_name not in {"typos", "weird", "angry"}:
            return raw_intent

        keyword_intent, _ = self._keyword_fallback_intent(user_message)
        if keyword_intent is not None:
            normalized_keyword = self._normalize_intent_name(keyword_intent)
            if normalized_keyword in self._actionable_intents or normalized_keyword in {
                "Missing Parcel", "Damaged Parcel", "Returns", "Shipping Cost",
                "Pickup", "International", "Payment", "Complaint", "General",
                "Address Change", "Delivery", "Business",
            }:
                return keyword_intent

        if lower_name in {"typos", "weird"}:
            lowered = str(user_message).lower()
            # Priority: compound patterns first (when+arrive, missing+received, etc.)
            if (re.search(r"\b(wen|when|wht|what|tim|time|date|day|soon|today|tomorrow|eta|estimat)\b", lowered)
                    and re.search(r"\b(arriv|arrive|delivr|deliver|delivry|reach|get|com|come)\b", lowered)):
                return "Delivery"
            if (re.search(r"\b(miss|lost|stole|stolen|nevr|never|didnt)\b.*\b(receiv|receive|arriv|arrive|delivr|deliver)\b", lowered)
                    or re.search(r"\b(not|didnt|never)\b.*\b(get|receiv|receive)\b", lowered)
                    or "没收到" in lowered or "未收到" in lowered or "找不到" in lowered or "丢了" in lowered):
                return "Missing Parcel"
            if any(kw in lowered for kw in ["damaged", "damagd", "broken", "broke", "crushed", "crushd", "破损", "损坏"]):
                return "Damaged Parcel"
            if (re.search(r"\b(pay|paymnt|payment|card|cash|reject|rejectd)\b", lowered)
                    or "支付" in lowered or "付款" in lowered or "扣款" in lowered):
                return "Payment"
            if (re.search(r"\b(addr|address|adress|addrss|postcod|postcode|office|recipient|recipt|apartmnt|apartment)\b", lowered)
                    or "地址" in lowered or "收货" in lowered or "改地址" in lowered):
                return "Address Change"
            if any(kw in lowered for kw in ["return", "rturn", "refund", "rfund", "cancel", "cancel"]):
                return "Returns"
            if (re.search(r"\b(ship|shipping|cost|price|discount|fre|free)\b.*\b(ship|shipping|cost|price|fee)\b", lowered)
                    or "运费" in lowered or "费用" in lowered):
                return "Shipping Cost"
            if any(kw in lowered for kw in ["pickup", "pick up", "pckup", "collect", "self pick", "自取"]):
                return "Pickup"
            if any(kw in lowered for kw in ["internation", "overseas", "customs", "custom", "import", "tax", "国际", "海关", "关税"]):
                return "International"
            if any(kw in lowered for kw in ["complain", "complaint", "terrible", "disappoint", "angry", "投诉", "不满意", "差劲"]):
                return "Complaint"
            if (re.search(r"\b(need|urgent|hurry|faster|quicker|express|priority|rush)\b.*\b(parcel|packag|delivery|delivr|ship|order)\b", lowered)
                    or re.search(r"\b(parcel|packag|delivery|delivr|ship|order)\b.*\b(need|urgent|hurry|faster|express|priority|rush)\b", lowered)):
                return "expedite_delivery"
            if (re.search(r"\b(wher|where|track|trck|trak|status|statu|location|updat|update|in transit|out for delivery)\b", lowered)
                    and re.search(r"\b(parcel|packag|pakage|order|ordr|shipment|shipmnt|delivry|delivery)\b", lowered)):
                return "Tracking"
            eta_kw = [
                "when", "wen", "arrive", "arriv", "time", "date", "day", "soon", "today", "tomorrow",
                "什么时候", "几时", "预计", "到达", "送到", "estimate", "eta",
            ]
            if any(kw in lowered for kw in eta_kw):
                return "Delivery"
            tracking_kw = [
                "parcel", "package", "parcl", "pakage", "pakkage", "order", "ordr",
                "shipment", "shipmnt", "track", "trck", "trak", "where", "wher",
                "status", "statu", "delivery", "delivry",
                "包裹", "快递", "在哪", "状态",
            ]
            if any(kw in lowered for kw in tracking_kw):
                return "Tracking"

        return raw_intent

    def _remember_user_input(self, user_message):
        """Store a short memory of recent user inputs and extracted context for the same session."""
        normalized = user_message.strip()
        if not normalized:
            return

        lower_input = normalized.lower()
        for previous in self.session_context["questions"]:
            if previous.lower() == lower_input:
                return

        self.session_context["questions"].append(normalized)
        if len(self.session_context["questions"]) > 25:
            self.session_context["questions"] = self.session_context["questions"][-25:]

        # Extract tracking IDs and remember recent order references.
        tracking_ids = re.findall(r'\bTRK\d{4}\b', normalized, re.IGNORECASE)
        for tracking_id in tracking_ids:
            tracking_id = tracking_id.upper()
            if tracking_id not in self.session_context["tracking_ids"]:
                self.session_context["tracking_ids"].append(tracking_id)
            if tracking_id not in self.session_context["recent_orders"]:
                self.session_context["recent_orders"].append(tracking_id)
            self.session_context["last_known_order"] = tracking_id
            if len(self.session_context["recent_orders"]) > 12:
                self.session_context["recent_orders"] = self.session_context["recent_orders"][-12:]

        # Extract address-like context when users mention a delivery address or destination.
        lower_text = normalized.lower()
        if any(keyword in lower_text for keyword in ["deliver to", "ship to", "address is", "new address", "send to", "office instead", "my office", "my home"]):
            addr_text = re.split(r'\b(?:deliver to|ship to|address is|new address|send to|office instead|my office|my home)\b', normalized, flags=re.IGNORECASE, maxsplit=1)
            if len(addr_text) > 1:
                address_value = addr_text[1].strip()
                if address_value and address_value not in self.session_context["addresses"]:
                    self.session_context["addresses"].append(address_value)

    def _get_memory_hint(self, user_message):
        """Return known prior user inputs for this chatbot session, if any."""
        return self.session_context["questions"]

    def _get_recent_orders_summary(self):
        if not self.session_context["recent_orders"]:
            return ""
        return ", ".join(self.session_context["recent_orders"][-5:])

    def _get_contextual_tracking_response(self, user_message, cleaned_message):
        """Use remembered tracking IDs when users ask for the status of a recently mentioned order.
        Now uses the unified natural summary builder (no raw DB labels)."""
        lower_message = cleaned_message.lower()
        tracking_context = self.session_context["tracking_ids"]
        if not tracking_context:
            return None

        wants_tracking = any(keyword in lower_message for keyword in [
            "where is my parcel", "where is my package", "track my", "where is my order",
            "parcel status", "tracking", "status", "delivery status", "still moving",
            "has my parcel", "what is the status"
        ])

        if wants_tracking:
            tracking_id = tracking_context[-1]
            if tracking_id in self.mock_db:
                order_info = self.mock_db[tracking_id]
                use_cn = self._contains_chinese(cleaned_message)
                memory_prefix_cn = f"我记得你之前提到过 {tracking_id}。"
                memory_prefix_en = f"I remember you previously mentioned {tracking_id}. "
                summary = self._build_natural_order_summary(order_info, tracking_id, focus='status', use_chinese=use_cn)
                return (memory_prefix_cn if use_cn else memory_prefix_en) + summary
        return None

    def _ml_intent_and_confidence(self, model_message):
        """TF-IDF + LinearSVC prediction with a decision score and class margin."""
        transformed_text = self.vectorizer.transform([model_message])
        predicted_intent = self.model.predict(transformed_text)[0]
        confidence, margin = linear_svc_decision_scores(self.model, transformed_text)
        return predicted_intent, float(confidence), float(margin)

    def _prediction_is_uncertain(self, confidence, margin):
        return confidence < self.threshold or margin < self.margin_threshold

    def get_bot_response(self, user_message):
        cleaned_message = user_message.strip()
        if not cleaned_message:
            return self._finalize_response(
                "Please say something, I am listening.",
                "empty_input",
                1.0,
                nlp_analysis={
                    "nlp_method": "Rule-based NLP",
                    "fallback_used": False,
                    "preprocessed_text": "",
                },
            )

        self._reload_orders_csv()
        self._remember_user_input(cleaned_message)

        tracking_in_message, order_in_message = self._extract_ids_from_text(cleaned_message)
        if tracking_in_message or order_in_message:
            self._remember_ids(tracking_id=tracking_in_message, order_id=order_in_message)

        semantic_intent = self._classify_semantic_intent(user_message, cleaned_message)
        last_intent = self.session_context.get('last_intent')

        if semantic_intent == 'affirmation':
            continued_intent = self._resolve_continued_intent()
            if continued_intent in self._actionable_intents:
                tracking_id, source = self._resolve_tracking_id(cleaned_message)
                reply, intent_name, conf = self._dispatch_logistics_intent(
                    continued_intent, user_message, cleaned_message, tracking_id, source
                )
                if reply is not None:
                    return self._finalize_response(
                        reply, continued_intent, conf, tracking_id=tracking_id, order_id=order_in_message,
                        nlp_analysis={
                            "nlp_method": "Rule-based NLP (conversation context)",
                            "fallback_used": False,
                            "preprocessed_text": preprocess_text(cleaned_message),
                        },
                    )
            reply = (
                "Yes, I can continue from our previous message. "
                "Please share the tracking number or tell me whether you want tracking, an arrival time, or faster delivery."
            )
            return self._finalize_response(
                reply, 'conversation_context', 0.9,
                nlp_analysis={
                    "nlp_method": "Rule-based NLP (conversation context)",
                    "fallback_used": False,
                    "preprocessed_text": preprocess_text(cleaned_message),
                },
            )

        # A tracking ID alone should continue the previous logistics request when one exists.
        if semantic_intent == 'order_summary' and last_intent in self._actionable_intents and last_intent != 'order_summary':
            semantic_intent = last_intent

        tracking_id, source = self._resolve_tracking_id(cleaned_message)

        if tracking_id is not None and semantic_intent is None:
            reply, focus_intent, conf = self._handle_tracking_only_query(
                user_message, cleaned_message, tracking_id, source=source or 'message'
            )
            if reply is not None:
                chosen = focus_intent if focus_intent in self._actionable_intents else 'Tracking'
                return self._finalize_response(
                    reply, chosen, conf or 0.96, tracking_id=tracking_id, order_id=order_in_message,
                    nlp_analysis={
                        "nlp_method": "Rule-based NLP (tracking ID presence + orders.csv lookup)",
                        "fallback_used": False,
                        "preprocessed_text": preprocess_text(cleaned_message),
                    },
                )

        if semantic_intent in self._actionable_intents:
            reply, intent_name, conf = self._dispatch_logistics_intent(
                semantic_intent, user_message, cleaned_message, tracking_id, source
            )
            if reply is not None:
                return self._finalize_response(
                    reply, semantic_intent, conf, tracking_id=tracking_id, order_id=order_in_message,
                    nlp_analysis={
                        "nlp_method": "Rule-based NLP (keyword / regex / entity detection)",
                        "fallback_used": False,
                        "preprocessed_text": preprocess_text(cleaned_message),
                    },
                )

        # Simple Chinese detection + translation fallback (if available)
        model_message = cleaned_message
        try:
            if self._contains_chinese(cleaned_message):
                model_message = self._translate_chinese_to_english(cleaned_message)
        except Exception:
            model_message = cleaned_message

        predicted_intent, max_confidence, decision_margin = self._ml_intent_and_confidence(model_message)
        normalized_intent = self._normalize_intent_name(predicted_intent)

        if str(normalized_intent).lower() in {"typos", "weird", "angry"}:
            disambiguated = self._disambiguate_similarity_result(
                normalized_intent, max_confidence, model_message
            )
            if str(disambiguated).lower() != str(normalized_intent).lower():
                normalized_intent = disambiguated
                max_confidence = max(max_confidence, 0.85)
        preprocessed = preprocess_text(model_message)
        ml_trace = {
            "ml_intent": str(predicted_intent),
            "ml_confidence": max_confidence,
            "decision_margin": decision_margin,
            "preprocessed_text": preprocessed,
        }

        keyword_intent, keyword_score = self._keyword_fallback_intent(model_message)

        # --- If semantic_intent is a clear non-actionable logistics intent
        # (Address Change, International, Damaged Parcel etc.), lock it in early
        # so a noisy LinearSVC prediction (e.g. tracking_request) can't override it.
        semantic_is_clear_logistics = (
            semantic_intent is not None
            and semantic_intent not in self._actionable_intents
            and self._normalize_intent_name(semantic_intent) in {
                "Address Change", "International", "Damaged Parcel", "Missing Parcel",
                "Returns", "Shipping Cost", "Pickup", "Payment", "Complaint",
                "General", "Delivery", "Business",
            }
        )
        if semantic_is_clear_logistics:
            normalized_intent = self._normalize_intent_name(semantic_intent)
            max_confidence = max(max_confidence, 0.90)

        if keyword_intent in self._actionable_intents:
            normalized_intent = keyword_intent
            max_confidence = max(max_confidence, float(keyword_score))
            reply, intent_name, conf = self._dispatch_logistics_intent(
                keyword_intent, user_message, cleaned_message, tracking_id, source
            )
            if reply is not None:
                return self._finalize_response(
                    reply, keyword_intent, max(conf or 0.9, float(keyword_score)), tracking_id=tracking_id,
                    nlp_analysis={
                        **ml_trace,
                        "nlp_method": "Rule-based NLP (keyword / regex) over LinearSVC",
                        "fallback_used": False,
                    },
                )

        if str(normalized_intent).lower() in {'weird', 'typos'} and keyword_intent in self._actionable_intents:
            reply, intent_name, conf = self._dispatch_logistics_intent(
                keyword_intent, user_message, cleaned_message, tracking_id, source
            )
            if reply is not None:
                return self._finalize_response(
                    reply, keyword_intent, 0.92, tracking_id=tracking_id,
                    nlp_analysis={
                        **ml_trace,
                        "nlp_method": "Rule-based NLP (keyword override)",
                        "fallback_used": True,
                        "fallback_method": "Rule-based keyword matching",
                    },
                )

        if str(normalized_intent).lower() == 'delivery' and self._is_expedite_query(user_message, cleaned_message):
            reply, intent_name, conf = self._dispatch_logistics_intent(
                'expedite_delivery', user_message, cleaned_message, tracking_id, source
            )
            if reply is not None:
                return self._finalize_response(
                    reply, 'expedite_delivery', conf or 0.95, tracking_id=tracking_id,
                    nlp_analysis={
                        **ml_trace,
                        "nlp_method": "Rule-based NLP (regex) over LinearSVC",
                        "fallback_used": False,
                    },
                )

        if str(normalized_intent).lower() == 'delivery' and self._is_eta_query(user_message, cleaned_message):
            reply, intent_name, conf = self._dispatch_logistics_intent(
                'Delivery', user_message, cleaned_message, tracking_id, source
            )
            if reply is not None:
                return self._finalize_response(
                    reply, 'Delivery', conf or 0.95, tracking_id=tracking_id,
                    nlp_analysis={
                        **ml_trace,
                        "nlp_method": "Rule-based NLP (regex) over LinearSVC",
                        "fallback_used": False,
                    },
                )

        if self._prediction_is_uncertain(max_confidence, decision_margin):
            if keyword_intent is not None:
                normalized_fallback = self._normalize_intent_name(keyword_intent)
                if normalized_fallback in self._actionable_intents:
                    reply, intent_name, conf = self._dispatch_logistics_intent(
                        normalized_fallback, user_message, cleaned_message, tracking_id, source
                    )
                    if reply is not None:
                        return self._finalize_response(
                            reply, normalized_fallback, float(keyword_score), tracking_id=tracking_id,
                            nlp_analysis={
                                **ml_trace,
                                "nlp_method": "TF-IDF + LinearSVC",
                                "fallback_used": True,
                                "fallback_method": "Rule-based keyword matching",
                            },
                        )
                reply = self._generate_nlp_reply(normalized_fallback, cleaned_message)
                return self._finalize_response(
                    reply, normalized_fallback, float(keyword_score), tracking_id=tracking_id,
                    nlp_analysis={
                        **ml_trace,
                        "nlp_method": "TF-IDF + LinearSVC",
                        "fallback_used": True,
                        "fallback_method": "Rule-based keyword matching",
                    },
                )

            fallback_intent, fallback_score = self._fallback_similarity_intent(model_message)
            if fallback_intent is not None:
                normalized_fallback = self._normalize_intent_name(fallback_intent)
                core = {"Tracking", "Delivery", "Address Change", "Damaged Parcel",
                        "Missing Parcel", "Returns", "expedite_delivery", "order_summary"}
                if normalized_fallback not in core and float(fallback_score) < 0.50:
                    normalized_fallback = "Weird"
                    fallback_score = max(float(fallback_score), 0.25)
                reply = self._generate_nlp_reply(normalized_fallback, cleaned_message)
                return self._finalize_response(
                    reply, normalized_fallback, float(fallback_score), tracking_id=tracking_id,
                    nlp_analysis={
                        **ml_trace,
                        "nlp_method": "TF-IDF + LinearSVC",
                        "fallback_used": True,
                        "fallback_method": "Cosine Similarity",
                        "similarity_score": float(fallback_score),
                    },
                )

            fallback_reply = random.choice(self.responses.get("default", ["I cannot understand your request clearly."]))
            recent_inputs = self._get_memory_hint(cleaned_message)
            if recent_inputs:
                recent_preview = ", ".join(recent_inputs[-5:])
                fallback_reply = fallback_reply + f" I remember you asked about these recent items: {recent_preview}."
            return self._finalize_response(
                fallback_reply, "unknown_fallback", max_confidence,
                nlp_analysis={
                    **ml_trace,
                    "nlp_method": "TF-IDF + LinearSVC",
                    "fallback_used": True,
                    "fallback_method": "Unknown / default template",
                },
            )

        reply = self._generate_nlp_reply(normalized_intent, cleaned_message)

        last_inputs = self._get_memory_hint(cleaned_message)
        if len(last_inputs) > 1:
            existing = last_inputs[:-1]
            if any(msg.lower() == cleaned_message.lower() for msg in existing):
                reply = reply + " You asked this before earlier in this chat, so I am reusing the latest relevant answer for you."

        return self._finalize_response(
            reply, normalized_intent, max_confidence, tracking_id=tracking_id,
            nlp_analysis={
                **ml_trace,
                "nlp_method": "TF-IDF + LinearSVC",
                "fallback_used": False,
            },
        )

    def get_bot_response_with_analysis(self, user_message):
        """Thin wrapper around ``get_bot_response`` that also returns the filled
        NLP analysis dict. Used by the Streamlit NLP Analysis and Dataset pages.

        Returns
        -------
        reply : str
            The final natural-language answer to the user.
        analysis : dict
            Copy of ``self.last_nlp_analysis`` with all per-interaction trace
            fields (detected_intent, nlp_method, decision_confidence,
            fallback_used, fallback_method, similarity_score, ml_intent,
            ml_confidence, decision_margin, preprocessed_text).
        """
        reply = self.get_bot_response(user_message)
        analysis = dict(self.last_nlp_analysis or self._empty_nlp_analysis())
        return reply, analysis

# (packaged-build persistence marker)

# =====================================================================
# Packaged-build feedback persistence (injected by build.py)
# =====================================================================
# Writes are always mirrored to the local CSV first. If either Google
# Sheets or GitHub secrets are configured, the same row is also written
# remotely, surviving Streamlit Cloud container restarts / cold boots.
#
# Configuration (Streamlit Cloud → App Settings → Secrets):
#
#   (A) Google Sheets  --------------------------------------------------
#   [connections.gsheets]
#   type = "gsheets"
#   spreadsheet = "https://docs.google.com/spreadsheets/d/XXXXXXXXX/edit"
#   worksheet = "user_feedback"
#   # Service account JSON (paste the entire GCP service-account object)
#   [connections.gsheets.secrets_account]
#   type = "service_account"
#   project_id = "...."
#   private_key_id = "...."
#   private_key = "-----BEGIN PRIVATE KEY-----\n....\n-----END PRIVATE KEY-----\n"
#   client_email = "xxxx@xxxx.iam.gserviceaccount.com"
#   client_id = "...."
#   auth_uri = "https://accounts.google.com/o/oauth2/auth"
#   token_uri = "https://oauth2.googleapis.com/token"
#   auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
#   client_x509_cert_url = "https://www.googleapis.com/robot/v1/metadata/x509/xxxx%40xxxx.iam.gserviceaccount.com"
#
#   → Remember to share the Sheet with the client_email above as "Editor"
#     and create a tab named exactly user_feedback with the CSV header row.
#
#   (B) GitHub CSV commit back to the repo  -----------------------------
#   [github]
#   token = "ghp_xxxxxxxxxxxxxxxxxxxx"        # Fine-grained PAT with "Contents: R/W" on your repo
#   owner = "your-github-username"
#   repo  = "your-repo-name"
#   branch = "main"
#   feedback_path = "data/user_feedback.csv"
#
# =====================================================================

_FEEDBACK_COLS = [
    "timestamp_utc",
    "user_message",
    "bot_reply",
    "detected_intent",
    "rating_1_5",
    "helpful_bool",
    "comment",
]


def _try_append_remote_gsheets(row: list) -> str | None:
    """Return None on success, otherwise a short error reason."""
    try:
        import streamlit as st  # noqa: WPS433  (optional — in this build only)
        from streamlit.connections import GSheetsConnection  # type: ignore
    except Exception:  # noqa: BLE001
        return "gsheets not available in this build"
    try:
        import streamlit as st  # noqa: WPS433
        conn = st.connection("gsheets", type=GSheetsConnection)
        import pandas as _pd
        try:
            existing = conn.read(worksheet="user_feedback", ttl=0)
        except Exception:  # noqa: BLE001
            existing = _pd.DataFrame(columns=_FEEDBACK_COLS)
        new_row = _pd.DataFrame([dict(zip(_FEEDBACK_COLS, row))])
        combined = _pd.concat([existing, new_row], ignore_index=True)
        conn.update(worksheet="user_feedback", data=combined.reset_index(drop=True))
        return None
    except Exception as exc:  # noqa: BLE001
        return f"gsheets: {type(exc).__name__}: {exc}"[:240]


def _try_append_remote_github(row: list) -> str | None:
    """Return None on success, otherwise a short error reason."""
    try:
        import streamlit as st  # noqa: WPS433
        gh = st.secrets.get("github", {}) if hasattr(st, "secrets") else {}
    except Exception:  # noqa: BLE001
        gh = {}
    if not gh or not gh.get("token") or not gh.get("repo"):
        return "github secrets not configured"
    try:
        import base64 as _b64
        import json as _json
        import requests as _rq
        token = gh["token"]
        owner = gh.get("owner") or gh.get("repo", "/").split("/")[0]
        repo = gh.get("repo").split("/")[-1] if "/" in gh.get("repo", "") else gh["repo"]
        branch = gh.get("branch") or "main"
        csv_path = gh.get("feedback_path") or "data/user_feedback.csv"
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        # Fetch current SHA and content for the file
        api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{csv_path}"
        resp = _rq.get(api_url, headers=headers, params={"ref": branch}, timeout=20)
        sha, existing_b64 = None, ""
        if resp.status_code == 200:
            payload = resp.json()
            sha = payload.get("sha")
            existing_b64 = payload.get("content", "") or ""
        elif resp.status_code != 404:
            return f"github GET {resp.status_code}"
        try:
            import csv as _csv
            existing_txt = (
                _b64.b64decode(existing_b64).decode("utf-8")
                if existing_b64 else ""
            )
            needs_header = (not existing_txt) or not existing_txt.strip()
            import io as _io
            buf = _io.StringIO(newline="")
            writer = _csv.writer(buf, lineterminator="\n")
            if needs_header:
                writer.writerow(_FEEDBACK_COLS)
            writer.writerow(row)
            chunk = buf.getvalue()
            if existing_txt and not existing_txt.endswith("\n"):
                existing_txt += "\n"
            new_txt = existing_txt + chunk
            new_b64 = _b64.b64encode(new_txt.encode("utf-8")).decode("ascii")
            message = "chore(user_feedback): append 1 row from chatbot session"
            body = {"message": message, "content": new_b64, "branch": branch}
            if sha:
                body["sha"] = sha
            put = _rq.put(api_url, headers=headers, data=_json.dumps(body), timeout=30)
            if put.status_code in (200, 201):
                return None
            return f"github PUT {put.status_code}: {put.text[:200]}"
        except Exception as exc_inner:  # noqa: BLE001
            return f"github encode: {type(exc_inner).__name__}: {exc_inner}"[:200]
    except Exception as exc:  # noqa: BLE001
        return f"github: {type(exc).__name__}: {exc}"[:240]


def _mirror_feedback_to_remotes(row: list) -> tuple[bool, str]:
    """Try all configured remote writers. Returns (any_success, human_report)."""
    any_success = False
    parts: list[str] = []
    # Google Sheets
    err_gs = _try_append_remote_gsheets(row)
    if err_gs is None:
        any_success = True
        parts.append("Google Sheets: OK")
    elif "not configured" in err_gs or "not available" in err_gs:
        pass  # silently skip — user hasn't set this backend up
    else:
        parts.append(f"Google Sheets: FAIL ({err_gs})")
    # GitHub CSV commit
    err_gh = _try_append_remote_github(row)
    if err_gh is None:
        any_success = True
        parts.append("GitHub: OK")
    elif "not configured" in err_gh:
        pass
    else:
        parts.append(f"GitHub: FAIL ({err_gh})")
    return any_success, " | ".join(parts) if parts else "no remote backend configured — local only"


def _persistence_status() -> dict:
    """Check which feedback backends are currently reachable / configured.

    Used by the chat sidebar to display the current persistence tier.
    """
    status = {"local_csv": True, "gsheets_configured": False, "github_configured": False}
    try:
        import streamlit as st  # noqa: WPS433
        try:
            st.connection("gsheets")
            status["gsheets_configured"] = True
        except Exception:  # noqa: BLE001
            pass
        try:
            gh = st.secrets.get("github", {})
            if gh.get("token") and gh.get("repo"):
                status["github_configured"] = True
        except Exception:  # noqa: BLE001
            pass
    except Exception:  # noqa: BLE001
        pass
    return status

