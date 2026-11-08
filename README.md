# NLP Assignment 1 - Data Acquisition, Text Processing and Cleaning

Phase 1 of the semester project: binary sentiment classification of IMDB movie reviews.

## Dataset

| Item | Value |
|---|---|
| Name | Large Movie Review Dataset (IMDB) - Maas et al., 2011 |
| Source | Zenodo record 7928582 (file `IMDB Dataset.csv`): https://zenodo.org/records/7928582 |
| Original release | https://ai.stanford.edu/~amaas/data/sentiment/ (used as automatic download if no CSV is given) |
| Size | 50,000 labelled reviews (25,000 positive, 25,000 negative); the 50,000 unlabelled reviews are not used |
| Licence | Not specified in the Zenodo record; the original release is for research use, cite Maas et al. (2011) |
| Target (Phase 2) | `sentiment`: positive / negative |

The raw data is never committed. `data/` is git-ignored and recreated by the scripts.

## Repository layout

```
run_all.py                  one command that runs everything
team.json                   team name, members, roll numbers, contributions (fill in)
requirements.txt
src/
  config.py                 paths and constants
  patterns.py               shared regular expressions
  download_dataset.py       downloads and parses the dataset into data/raw/imdb_reviews.csv
  explore.py                statistics, Zipf plot, messy-text examples
  tokenization.py           4 tokenizers compared, regex patterns, challenges
  clean_text.py             the cleaning pipeline (clean_text) and output writer
  build_report.py           builds reports/assignment1_report.pdf from measured results
notebooks/exploration.ipynb exploration notebook
tests/test_clean_text.py    unit tests for the cleaning function
```

## Run from a fresh clone

Requires Python 3.11 or newer.

```bash
git clone https://github.com/imrahhassan12-netizen/NLP_Assignment_1.git
cd NLP_Assignment_1
git checkout assignment-1

python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python run_all.py
```

Download `IMDB Dataset.csv` from the Zenodo record above, then run:

```bash
python run_all.py --from-csv "IMDB Dataset.csv"
```

Individual stages can be run on their own:

```bash
python -m src.download_dataset
python -m src.explore
python -m src.tokenization
python -m src.clean_text
python -m src.build_report
python -m unittest discover -s tests -t .
```

NLTK's `stopwords` list is downloaded automatically on first use.

## Outputs

| File | Description |
|---|---|
| `data/processed/cleaned_reviews.csv` | columns `cleaned_review`, `sentiment`; loaded directly in Phase 2 |
| `reports/assignment1_report.pdf` | the report, filled with the measured results |
| `reports/stats.json` | every number used in the report |
| `reports/logs/*.txt` | console output of each stage |

## Cleaning pipeline (order matters)

1. Unicode NFKC normalisation
2. HTML entity decoding and tag removal
3. URL and e-mail removal
4. Rating normalisation (`7/10` becomes `rating7`)
5. Lowercasing
6. Contraction expansion (`n't` becomes `not`, `'ll` becomes `will`, ...)
7. Regex tokenization
8. Stop-word removal (NLTK list; `no`, `nor`, `not`, `never`, `very`, `too`, `but`, `against`, `only`, `few`, `more`, `most` are kept)
9. Porter stemming

## Team

- Imrah Maqsood - Roll No: 0293
- Ajwa

Repository: https://github.com/imrahhassan12-netizen/NLP_Assignment_1

Team details live in `team.json`. After editing it, run `python -m src.build_report` to regenerate the PDF.

## Git workflow

Use one branch per task and open pull requests into `main`:

```bash
git checkout -b feature/cleaning-pipeline
git add .
git commit -m "Add cleaning pipeline"
git push -u origin feature/cleaning-pipeline
```

When everything is merged:

```bash
git tag assignment-1
git push origin assignment-1
```
