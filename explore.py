import platform
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import FIG_DIR, SAMPLE_SIZE, SEED
from src.patterns import (
    ALL_CAPS, HTML_TAG, IRREGULAR_SPACE, NON_ASCII, RATING, REPEATED_PUNCT, URL,
)
from src.utils import load_raw, save_stats, words

MESSY_PATTERNS = {
    "HTML line-break tags": HTML_TAG,
    "URLs": URL,
    "ALL-CAPS emphasis": ALL_CAPS,
    "Repeated punctuation": REPEATED_PUNCT,
    "Non-ASCII characters": NON_ASCII,
    "Irregular spacing": IRREGULAR_SPACE,
    "Rating expressions": RATING,
}


def snippet(text, match, width=55):
    start = max(0, match.start() - width)
    return " ".join(text[start:match.end() + width].split())


def messy_examples(df):
    examples = []
    for issue, pattern in MESSY_PATTERNS.items():
        hits = 0
        first = None
        for text in df["review"]:
            match = pattern.search(text)
            if match:
                hits += 1
                if first is None:
                    first = snippet(text, match)
        examples.append(
            {
                "issue": issue,
                "reviews_affected": hits,
                "share_pct": round(100 * hits / len(df), 2),
                "example": first or "",
            }
        )
    return examples


def zipf_plot(counter):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    freqs = sorted(counter.values(), reverse=True)
    fig, ax = plt.subplots(figsize=(5.2, 3.0), dpi=150)
    ax.loglog(range(1, len(freqs) + 1), freqs, linewidth=1.4)
    ax.set_xlabel("Rank (log)")
    ax.set_ylabel("Frequency (log)")
    ax.set_title("Rank-frequency plot of the 1,000-review sample")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "zipf.png")
    plt.close(fig)


def main():
    df = load_raw()
    lengths = df["review"].str.split().str.len()
    target = df["sentiment"].value_counts()

    sample = df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=SEED)
    counter = Counter(w for text in sample["review"] for w in words(text))
    total_tokens = sum(counter.values())
    types = len(counter)
    hapax = sum(1 for c in counter.values() if c == 1)
    top100 = sum(c for _, c in counter.most_common(100))
    zipf_plot(counter)

    import nltk
    import pandas
    import spacy

    stats = {
        "n_documents": int(len(df)),
        "avg_words": round(float(lengths.mean()), 2),
        "median_words": float(lengths.median()),
        "min_words": int(lengths.min()),
        "max_words": int(lengths.max()),
        "target_counts": {k: int(v) for k, v in target.items()},
        "target_pct": {k: round(100 * v / len(df), 2) for k, v in target.items()},
        "sample_size": int(len(sample)),
        "sample_tokens": total_tokens,
        "sample_types": types,
        "sample_ttr": round(types / total_tokens, 4),
        "sample_hapax_pct": round(100 * hapax / types, 1),
        "sample_top100_pct": round(100 * top100 / total_tokens, 1),
        "top_words": counter.most_common(12),
        "messy_examples": messy_examples(df),
        "environment": {
            "python": platform.python_version(),
            "pandas": pandas.__version__,
            "nltk": nltk.__version__,
            "spacy": spacy.__version__,
        },
    }
    save_stats("exploration", stats)

    print(f"Documents            : {stats['n_documents']:,}")
    print(f"Average length       : {stats['avg_words']} words "
          f"(min {stats['min_words']}, median {stats['median_words']:.0f}, max {stats['max_words']:,})")
    print("Target distribution  :", stats["target_counts"])
    print(f"Sample ({stats['sample_size']} reviews): tokens={total_tokens:,}, "
          f"types={types:,}, TTR={stats['sample_ttr']}")
    print(f"Hapax legomena       : {stats['sample_hapax_pct']}% of types")
    print(f"Top-100 words cover  : {stats['sample_top100_pct']}% of tokens")
    print("Most frequent words  :", ", ".join(f"{w}({c})" for w, c in stats["top_words"][:8]))
    print("Messy-text findings  :")
    for row in stats["messy_examples"]:
        print(f"  {row['issue']:<22} {row['reviews_affected']:>6,} reviews ({row['share_pct']}%)")


if __name__ == "__main__":
    main()
