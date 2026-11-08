import html
import re
import unicodedata
from functools import lru_cache

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from src.config import CLEAN_FILE, PROCESSED_DIR
from src.patterns import EMAIL, HTML_TAG, RATING, URL
from src.utils import load_raw, save_stats, words

KEEP_WORDS = {
    "no", "nor", "not", "never", "very", "too", "but",
    "against", "only", "few", "more", "most",
}
APOSTROPHES = re.compile(r"[\u2018\u2019\u00b4`]")
TOKEN = re.compile(r"[a-z]+\d*|\d+")
CONTRACTIONS = [
    (re.compile(r"\bcan't\b"), "can not"),
    (re.compile(r"\bwon't\b"), "will not"),
    (re.compile(r"\bain't\b"), "is not"),
    (re.compile(r"n't\b"), " not"),
    (re.compile(r"'ll\b"), " will"),
    (re.compile(r"'re\b"), " are"),
    (re.compile(r"'ve\b"), " have"),
    (re.compile(r"'m\b"), " am"),
    (re.compile(r"'d\b"), " would"),
    (re.compile(r"'s\b"), ""),
]
STEM_DEMO_WORDS = [
    "watching", "watched", "movies", "acting", "loved", "loving",
    "disappointing", "disappointed", "terrible", "wonderfully",
]

_stemmer = PorterStemmer()


def _load_stopwords():
    try:
        base = set(stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        base = set(stopwords.words("english"))
    return base - KEEP_WORDS


STOPWORDS = _load_stopwords()


@lru_cache(maxsize=None)
def stem(word):
    return _stemmer.stem(word)


def clean_text(text):
    text = unicodedata.normalize("NFKC", str(text))
    text = html.unescape(text)
    text = HTML_TAG.sub(" ", text)
    text = URL.sub(" ", text)
    text = EMAIL.sub(" ", text)
    text = RATING.sub(lambda m: f" rating{m.group(1)} ", text)
    text = text.lower()
    text = APOSTROPHES.sub("'", text)
    for pattern, replacement in CONTRACTIONS:
        text = pattern.sub(replacement, text)
    tokens = TOKEN.findall(text)
    return " ".join(stem(t) for t in tokens if t not in STOPWORDS)


def vocabulary(token_lists):
    vocab = set()
    for tokens in token_lists:
        vocab.update(tokens)
    return vocab


def main():
    df = load_raw()
    df["cleaned_review"] = df["review"].apply(clean_text)

    raw_case = vocabulary(re.findall(r"\w+", str(t)) for t in df["review"])
    raw_lower = vocabulary(words(t) for t in df["review"])
    cleaned = vocabulary(t.split() for t in df["cleaned_review"])

    empty = int((df["cleaned_review"].str.len() == 0).sum())
    out = df.loc[df["cleaned_review"].str.len() > 0, ["cleaned_review", "sentiment"]]
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(CLEAN_FILE, index=False, encoding="utf-8")

    example_idx = int(df["review"].str.contains("<br", regex=False).idxmax())
    stats = {
        "vocab_raw_case_sensitive": len(raw_case),
        "vocab_raw_lowercase": len(raw_lower),
        "vocab_cleaned": len(cleaned),
        "reduction_vs_raw_pct": round(100 * (1 - len(cleaned) / len(raw_case)), 1),
        "reduction_vs_lowercase_pct": round(100 * (1 - len(cleaned) / len(raw_lower)), 1),
        "rows_in": int(len(df)),
        "rows_out": int(len(out)),
        "rows_dropped_empty": empty,
        "stopword_count": len(STOPWORDS),
        "kept_words": sorted(KEEP_WORDS),
        "stem_examples": [[w, stem(w)] for w in STEM_DEMO_WORDS],
        "example_before": df["review"].iloc[example_idx][:400],
        "example_after": df["cleaned_review"].iloc[example_idx][:400],
    }
    save_stats("cleaning", stats)

    print("Cleaning completed")
    print(f"Rows in / out        : {stats['rows_in']:,} / {stats['rows_out']:,}")
    print(f"Vocabulary (raw)     : {stats['vocab_raw_case_sensitive']:,}")
    print(f"Vocabulary (lowercase raw): {stats['vocab_raw_lowercase']:,}")
    print(f"Vocabulary (cleaned) : {stats['vocab_cleaned']:,}")
    print(f"Reduction vs raw     : {stats['reduction_vs_raw_pct']}%")
    print("Stem examples        :", ", ".join(f"{a}->{b}" for a, b in stats["stem_examples"]))
    print(f"Saved                : {CLEAN_FILE}")


if __name__ == "__main__":
    main()
