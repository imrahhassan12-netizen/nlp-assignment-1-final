import argparse
import contextlib
import io
import sys

from src import build_report, clean_text, download_dataset, explore, tokenization
from src.config import LOG_DIR


def run_stage(name, func, *args):
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        func(*args)
    output = buffer.getvalue()
    print(f"[{name}]")
    print(output)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    (LOG_DIR / f"{name}.txt").write_text(output, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-csv", help="use an existing IMDB CSV instead of downloading")
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args()

    run_stage("download_dataset", download_dataset.main, args.from_csv, args.force_download)
    run_stage("explore", explore.main)
    run_stage("tokenization", tokenization.main)
    run_stage("clean_text", clean_text.main)
    build_report.build()


if __name__ == "__main__":
    sys.exit(main())
