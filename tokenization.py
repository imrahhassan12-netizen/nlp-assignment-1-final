import re

import spacy
from nltk.tokenize import NLTKWordTokenizer

from src.config import SAMPLE_SIZE, SEED
from src.patterns import (
    EMAIL, HTML_TAG, HYPHENATED, NEGATION_CONTRACTION, NON_ASCII,
    RATING, REPEATED_PUNCT, URL,
)
from src.utils import load_raw, save_stats

SENTENCES = [
    "I couldn't believe it; the acting was top-notch, but the plot isn't great.<br /><br />Rated 7.5/10!",
    "Check www.imdb.com for the cast's full list... they're amazing!!!",
    "It's a feel-good movie (1994) that you'll love \U0001F60D",
]

REGEX_TABLE = [
    ("HTML tags", HTML_TAG, "Matches '<', any run of characters that is not '>', then '>'. Finds <br /> line breaks so they can be deleted instead of becoming a 'br' token."),
    ("URLs", URL, "Matches http(s):// or www. followed by non-space characters. Web links carry no sentiment."),
    ("Rating expressions", RATING, "Matches 1-2 digits, an optional decimal, optional spaces and '/10'. Keeps scores such as 7/10 as a single useful token (rating7)."),
    ("Repeated punctuation", REPEATED_PUNCT, "Matches a !, ? or . repeated 3+ times via a back-reference. Marks emphatic style."),
    ("Negation contractions", NEGATION_CONTRACTION, "Matches word characters followed by n't. Locates words such as didn't and isn't whose negation must survive cleaning."),
]

_nltk = NLTKWordTokenizer()
_spacy = spacy.blank("en")

TOKENIZERS = {
    "Whitespace split": lambda t: t.split(),
    "Regex \\b\\w+\\b": lambda t: re.findall(r"\b\w+\b", t),
    "NLTK (Treebank-style)": lambda t: _nltk.tokenize(t),
    "spaCy (English rules)": lambda t: [tok.text for tok in _spacy.tokenizer(t)],
}


def main():
    df = load_raw()
    sample = df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=SEED)

    sentence_results = [
        {"sentence": s, "tokens": {name: fn(s) for name, fn in TOKENIZERS.items()}}
        for s in SENTENCES
    ]

    corpus_counts = {}
    for name, fn in TOKENIZERS.items():
        tokens = [tok.lower() for text in sample["review"] for tok in fn(text)]
        corpus_counts[name] = {"tokens": len(tokens), "types": len(set(tokens))}

    regex_rows = []
    for name, pattern, explanation in REGEX_TABLE:
        hits = 0
        example = ""
        for text in df["review"]:
            match = pattern.search(text)
            if match:
                hits += 1
                if not example:
                    example = match.group(0)
        regex_rows.append(
            {
                "name": name,
                "pattern": pattern.pattern,
                "explanation": explanation,
                "reviews_matching": hits,
                "example": example,
            }
        )

    challenge_checks = {
        "Contractions (n't)": NEGATION_CONTRACTION,
        "Hyphenated words": HYPHENATED,
        "Non-ASCII / emoji characters": NON_ASCII,
        "HTML tags": HTML_TAG,
        "Email addresses": EMAIL,
    }
    challenges = {
        name: int(sum(1 for text in df["review"] if pat.search(text)))
        for name, pat in challenge_checks.items()
    }

    save_stats(
        "tokenization",
        {
            "sentences": sentence_results,
            "corpus_counts": corpus_counts,
            "regex_table": regex_rows,
            "challenge_review_counts": challenges,
        },
    )

    print("Tokenizer comparison on sample sentence 1:")
    for name, tokens in sentence_results[0]["tokens"].items():
        print(f"  {name:<24} {tokens}")
    print(f"Corpus counts on {len(sample)}-review sample:")
    for name, c in corpus_counts.items():
        print(f"  {name:<24} tokens={c['tokens']:,} types={c['types']:,}")
    print("Regex pattern matches (reviews):")
    for row in regex_rows:
        print(f"  {row['name']:<22} {row['reviews_matching']:,}")


if __name__ == "__main__":
    main()
