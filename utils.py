import json
import re

import pandas as pd

from src.config import RAW_FILE, STATS_FILE

WORD_RE = re.compile(r"\w+")


def load_raw():
    if not RAW_FILE.exists():
        raise FileNotFoundError(
            f"{RAW_FILE} not found. Run: python -m src.download_dataset"
        )
    return pd.read_csv(RAW_FILE)


def load_stats():
    if STATS_FILE.exists():
        return json.loads(STATS_FILE.read_text(encoding="utf-8"))
    return {}


def save_stats(key, value):
    stats = load_stats()
    stats[key] = value
    STATS_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATS_FILE.write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def words(text):
    return WORD_RE.findall(str(text).lower())
