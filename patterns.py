import re

HTML_TAG = re.compile(r"<[^>]+>")
URL = re.compile(r"https?://\S+|www\.\S+")
EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
RATING = re.compile(r"\b(\d{1,2})(?:\.\d)?\s*/\s*10\b")
REPEATED_PUNCT = re.compile(r"([!?.])\1{2,}")
NEGATION_CONTRACTION = re.compile(r"\b\w+n't\b", re.IGNORECASE)
HYPHENATED = re.compile(r"\b[A-Za-z]+(?:-[A-Za-z]+)+\b")
NON_ASCII = re.compile(r"[^\x00-\x7f]")
ALL_CAPS = re.compile(r"\b[A-Z]{4,}\b")
IRREGULAR_SPACE = re.compile(r"[ \t]{2,}|\s+[,.!?;:]")
