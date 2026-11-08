import argparse
import re
import tarfile
import urllib.request

import pandas as pd

from src.config import ARCHIVE_FILE, DATASET_URL, RAW_DIR, RAW_FILE

MEMBER_RE = re.compile(r"aclImdb/(train|test)/(pos|neg)/[^/]+\.txt$")


def read_archive(path):
    rows = []
    with tarfile.open(path, "r:gz") as tar:
        for member in tar:
            match = MEMBER_RE.search(member.name)
            if match and member.isfile():
                text = tar.extractfile(member).read().decode("utf-8")
                label = "positive" if match.group(2) == "pos" else "negative"
                rows.append((member.name, text, label))
    rows.sort()
    return pd.DataFrame(
        [(text, label) for _, text, label in rows], columns=["review", "sentiment"]
    )


def read_csv(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()
    df = df[["review", "sentiment"]].dropna()
    df["review"] = df["review"].astype(str)
    df["sentiment"] = df["sentiment"].str.strip().str.lower()
    return df.reset_index(drop=True)


def download_archive():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if not ARCHIVE_FILE.exists():
        print(f"Downloading {DATASET_URL}")
        urllib.request.urlretrieve(DATASET_URL, ARCHIVE_FILE)
    return ARCHIVE_FILE


def main(from_csv=None, force=False):
    if RAW_FILE.exists() and not force:
        print(f"Raw dataset already present: {RAW_FILE}")
        return
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df = read_csv(from_csv) if from_csv else read_archive(download_archive())
    df.to_csv(RAW_FILE, index=False, encoding="utf-8")
    print(f"Saved {len(df):,} reviews to {RAW_FILE}")
    print("Sentiment distribution:")
    print(df["sentiment"].value_counts().to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-csv", help="use an existing IMDB CSV instead of downloading")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    main(args.from_csv, args.force)
