from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports"
FIG_DIR = REPORT_DIR / "figures"
LOG_DIR = REPORT_DIR / "logs"

RAW_FILE = RAW_DIR / "imdb_reviews.csv"
ARCHIVE_FILE = RAW_DIR / "aclImdb_v1.tar.gz"
CLEAN_FILE = PROCESSED_DIR / "cleaned_reviews.csv"
STATS_FILE = REPORT_DIR / "stats.json"
TEAM_FILE = ROOT / "team.json"
REPORT_FILE = REPORT_DIR / "assignment1_report.pdf"

DATASET_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
SEED = 42
SAMPLE_SIZE = 1000
