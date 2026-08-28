"""
Shared NLP preprocessing for LogiBot.

Used during training, holdout testing, and live user prediction so the
TF-IDF vocabulary always sees the same token form.
"""

from __future__ import annotations

import re
import unicodedata

# Domain-safe stop words for short logistics questions.
# sklearn's full English list removes where/when/what/why/how/not, which
# are often the only signal that distinguishes Tracking from ETA.
DOMAIN_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "into", "over", "after", "is", "are", "was",
    "were", "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "can", "this", "that", "these", "those",
    "i", "you", "he", "she", "it", "we", "they", "me", "your", "our", "their",
    "please", "pls", "u",
}

_NLTK_READY = False
_LEMMATIZER = None


def _ensure_nltk():
    """Load NLTK resources once. Falls back gracefully if NLTK data is missing."""
    global _NLTK_READY, _LEMMATIZER

    if _NLTK_READY:
        return

    try:
        import nltk
        from nltk.stem import WordNetLemmatizer

        resources = [
            ("tokenizers/punkt", "punkt"),
            ("tokenizers/punkt_tab", "punkt_tab"),
            ("corpora/wordnet", "wordnet"),
            ("corpora/omw-1.4", "omw-1.4"),
        ]
        for path, name in resources:
            try:
                nltk.data.find(path)
            except LookupError:
                nltk.download(name, quiet=True)

        _LEMMATIZER = WordNetLemmatizer()
        _NLTK_READY = True
    except Exception:
        _LEMMATIZER = None
        _NLTK_READY = True


def clean_text(text) -> str:
    """Lowercase, strip URLs/punctuation noise, and collapse whitespace. Keeps CJK characters."""
    if text is None:
        return ""

    cleaned = unicodedata.normalize("NFKC", str(text))
    cleaned = cleaned.lower()
    cleaned = re.sub(r"https?://\S+|www\.\S+", " ", cleaned)
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    cleaned = re.sub(r"[^a-z0-9\u4e00-\u9fff\s'-]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _tokenize(text: str):
    _ensure_nltk()
    chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
    latin_chars = len(re.findall(r"[a-z]", text))

    # Character-level Chinese matching is weak in this English TF-IDF model.
    # Keep the cleaned phrase as one token so exact training phrases can still match.
    if chinese_chars >= 2 and chinese_chars > latin_chars:
        return [text] if text else []

    try:
        from nltk.tokenize import word_tokenize

        tokens = word_tokenize(text)
    except Exception:
        tokens = text.split()

    return [tok for tok in tokens if tok and tok not in {"'", "-"}]


def _lemmatize_token(token: str) -> str:
    _ensure_nltk()
    if _LEMMATIZER is None:
        return token
    if re.search(r"[\u4e00-\u9fff]", token):
        return token
    if not re.search(r"[a-z]", token):
        return token
    try:
        lemmatized = _LEMMATIZER.lemmatize(token, pos="v")
        lemmatized = _LEMMATIZER.lemmatize(lemmatized, pos="n")
        return lemmatized
    except Exception:
        return token


def tokenize_and_normalize(text) -> list:
    """
    Tokenize cleaned text, drop English stop words, and lemmatize.

    Stemming is intentionally not applied. Lemmatization is the only
    morphological step so tokens stay readable for TF-IDF n-grams.
    """
    _ensure_nltk()
    cleaned = clean_text(text)
    if not cleaned:
        return []

    tokens = []
    for token in _tokenize(cleaned):
        lemmatized = _lemmatize_token(token)
        if lemmatized in DOMAIN_STOPWORDS:
            continue
        if len(lemmatized) == 1 and lemmatized.isascii() and lemmatized.isalpha():
            continue
        tokens.append(lemmatized)
    return tokens


def preprocess_text(text) -> str:
    """Return a cleaned, lemmatized string. Used for inspection and cosine inputs."""
    return " ".join(tokenize_and_normalize(text))


def build_tfidf_vectorizer(ngram_range=(1, 2), max_features=8000):
    """TfidfVectorizer with the shared tokenizer so train/test/live input stay aligned."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    return TfidfVectorizer(
        preprocessor=clean_text,
        tokenizer=tokenize_and_normalize,
        token_pattern=None,
        lowercase=False,
        # Stop words are removed in tokenize_and_normalize. sklearn's built-in
        # 'english' list is not used because it drops where/when/what/how.
        stop_words=None,
        ngram_range=ngram_range,
        max_features=max_features,
        min_df=1,
    )
