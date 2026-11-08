import json
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image, ListFlowable, ListItem, Paragraph, Preformatted, SimpleDocTemplate,
    Spacer, Table, TableStyle,
)

from src.config import FIG_DIR, LOG_DIR, REPORT_FILE, TEAM_FILE
from src.utils import load_stats

BASE = getSampleStyleSheet()
BODY = ParagraphStyle("body", parent=BASE["BodyText"], fontSize=9, leading=11.5, spaceAfter=3)
CELL = ParagraphStyle("cell", parent=BODY, fontSize=7.8, leading=9.6, spaceAfter=0)
H1 = ParagraphStyle("h1", parent=BASE["Heading2"], fontSize=12, spaceBefore=7, spaceAfter=3, keepWithNext=1)
H2 = ParagraphStyle("h2", parent=BASE["Heading3"], fontSize=9.6, spaceBefore=4, spaceAfter=2, keepWithNext=1)
TITLE = ParagraphStyle("title", parent=BASE["Title"], fontSize=16, leading=19, spaceAfter=2)
SUB = ParagraphStyle("sub", parent=BODY, alignment=TA_CENTER, fontSize=9.5)
MONO = ParagraphStyle("mono", fontName="Courier", fontSize=6.6, leading=7.8)


def safe(value):
    text = str(value)
    text = "".join(c if ord(c) < 256 else f"<U+{ord(c):04X}>" for c in text)
    return escape(text)


def para(text, style=BODY):
    return Paragraph(text, style)


def cell(text):
    return Paragraph(safe(text), CELL)


def grid(rows, widths, header=True):
    data = [[cell(c) if not hasattr(c, "wrap") else c for c in row] for row in rows]
    table = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    style = [
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#999999")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]
    if header:
        style.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E6ECF5")))
    table.setStyle(TableStyle(style))
    return table


def bullets(items):
    return ListFlowable(
        [ListItem(para(i), leftIndent=10) for i in items],
        bulletType="bullet", bulletFontSize=6, leftIndent=12,
    )


def log_block(name, max_lines=9):
    path = LOG_DIR / f"{name}.txt"
    if not path.exists():
        return Preformatted("(log not found - run python run_all.py)", MONO)
    lines = path.read_text(encoding="utf-8").splitlines()[:max_lines]
    clipped = [safe(l)[:128] for l in lines]
    return Preformatted("\n".join(clipped), MONO)


def fmt(n):
    return f"{n:,}"


def build():
    stats = load_stats()
    ex, tk, cl = stats["exploration"], stats["tokenization"], stats["cleaning"]
    team = json.loads(TEAM_FILE.read_text(encoding="utf-8"))
    env = ex["environment"]
    W = A4[0] - 3.2 * cm
    s = []

    s.append(para("Assignment 1 - Project Phase 1", TITLE))
    s.append(para("Data Acquisition, Text Processing &amp; Cleaning | Introduction to NLP", SUB))
    s.append(para(f"<b>Team:</b> {safe(team['team_name'])} &nbsp;|&nbsp; <b>Repository:</b> "
                  f"{safe(team['repository_url'])} (tag <b>{safe(team['tag'])}</b>)", SUB))
    s.append(Spacer(1, 3))
    s.append(grid([["Member", "Roll number"]] + [[m["name"], m["roll_number"]] for m in team["members"]],
                  [W * 0.6, W * 0.4]))

    s.append(para("1. Topic and Dataset", H1))
    s.append(para(
        "<b>Topic:</b> binary sentiment classification of movie reviews (Phase 2 will train classifiers on the cleaned text). "
        "<b>Why this dataset:</b> it is a public, well-known benchmark with a large amount of free-form English text, "
        "perfectly balanced labels, and real-world noise (HTML, contractions, ratings), so every cleaning decision has a visible effect."))
    s.append(grid([
        ["Item", "Value"],
        ["Dataset", "Large Movie Review Dataset (IMDB), Maas et al., 2011"],
        ["Source", "Zenodo record 7928582 (IMDB Dataset.csv); original release by Maas et al. (2011), Stanford"],
        ["Size", f"{fmt(ex['n_documents'])} labelled reviews (train + test splits combined; unlabelled reviews excluded)"],
        ["Target", "sentiment: positive / negative"],
        ["Licence", "Not specified in the Zenodo record; the original release is for research use with citation of Maas et al. (2011)."],
        ["Environment", f"Python {env['python']}, pandas {env['pandas']}, NLTK {env['nltk']}, spaCy {env['spacy']}; virtual environment + requirements.txt"],
        ["Raw data", "Loaded into data/raw by src/download_dataset.py; never committed (data/ is git-ignored)"],
    ], [W * 0.17, W * 0.83]))

    s.append(para("2. Exploration and Basic Statistics", H1))
    s.append(grid([
        ["Statistic", "Value"],
        ["Documents", fmt(ex["n_documents"])],
        ["Average / median length (words)", f"{ex['avg_words']} / {ex['median_words']:.0f}"],
        ["Shortest / longest review (words)", f"{ex['min_words']} / {fmt(ex['max_words'])}"],
        ["Target distribution", ", ".join(f"{k}: {fmt(v)} ({ex['target_pct'][k]}%)" for k, v in ex["target_counts"].items())],
        [f"Sample of {fmt(ex['sample_size'])} reviews: tokens / types",
         f"{fmt(ex['sample_tokens'])} / {fmt(ex['sample_types'])} (type-token ratio {ex['sample_ttr']})"],
        ["Hapax legomena (types seen once)", f"{ex['sample_hapax_pct']}% of types"],
        ["Share of tokens covered by the 100 most frequent words", f"{ex['sample_top100_pct']}%"],
    ], [W * 0.45, W * 0.55]))
    s.append(Spacer(1, 3))
    zipf = FIG_DIR / "zipf.png"
    top = ", ".join(f"{safe(w)} ({fmt(c)})" for w, c in ex["top_words"][:8])
    s.append(para(
        f"<b>Zipf / long tail.</b> Most frequent words: {top}. "
        f"The log-log rank-frequency curve is close to a straight line: a few function words dominate "
        f"(the top 100 words cover {ex['sample_top100_pct']}% of all tokens) while {ex['sample_hapax_pct']}% of the vocabulary "
        f"occurs only once. This long tail of rare words is why normalisation (case folding, stemming) matters for a bag-of-words model. "
        f"The raw top list also shows markup noise such as 'br'."))
    if zipf.exists():
        s.append(Image(str(zipf), width=W * 0.62, height=W * 0.62 * 3.0 / 5.2))
    s.append(para("Examples of messy text found in the data", H2))
    s.append(grid(
        [["Issue", "Reviews affected", "Real example"]]
        + [[r["issue"], f"{fmt(r['reviews_affected'])} ({r['share_pct']}%)", r["example"][:150]]
           for r in ex["messy_examples"]],
        [W * 0.19, W * 0.15, W * 0.66]))

    s.append(para("3. Tokenization and Regular Expressions", H1))
    s.append(para("Four tokenizers were compared: whitespace split, a regex tokenizer, NLTK's Treebank-style tokenizer and spaCy's English tokenizer rules."))
    names = list(tk["sentences"][0]["tokens"].keys())
    rows = [["Input"] + [x["sentence"] for x in tk["sentences"]]]
    for name in names:
        rows.append([name] + [" | ".join(x["tokens"][name]) for x in tk["sentences"]])
    s.append(grid(rows, [W * 0.14] + [W * 0.86 / len(tk["sentences"])] * len(tk["sentences"])))
    s.append(Spacer(1, 3))
    s.append(grid(
        [["Tokenizer", f"Tokens ({ex['sample_size']}-review sample)", "Types (lowercased)"]]
        + [[n, fmt(c["tokens"]), fmt(c["types"])] for n, c in tk["corpus_counts"].items()],
        [W * 0.34, W * 0.36, W * 0.30]))
    s.append(para(
        "Whitespace splitting leaves punctuation glued to words ('great.&lt;br', 'amazing!!!'). The regex tokenizer removes punctuation but "
        "breaks contractions and hyphenated words apart (couldn't becomes couldn + t). NLTK and spaCy keep punctuation as separate tokens and split "
        "contractions into a stem plus a clitic (could + n't), which keeps the negation recoverable. None of them understands HTML (the NLTK output shows '&lt;', 'br', '/', '&gt;'), so markup must be removed before tokenizing."))
    s.append(para("Regular expressions written for this data", H2))
    s.append(grid(
        [["Name", "Pattern", "Explanation", "Reviews matching", "Example"]]
        + [[r["name"], r["pattern"], r["explanation"], fmt(r["reviews_matching"]), r["example"][:40]]
           for r in tk["regex_table"]],
        [W * 0.12, W * 0.2, W * 0.42, W * 0.1, W * 0.16]))
    ch = tk["challenge_review_counts"]
    n_contr = ch["Contractions (n't)"]
    s.append(para(
        f"<b>Tokenization challenges in this data.</b> (1) <b>Contractions and negation:</b> {fmt(n_contr)} reviews contain n't forms; "
        f"splitting them carelessly leaves stray tokens such as 't' or 'll' and can lose 'not', which flips sentiment. "
        f"(2) <b>Markup and hyphenation:</b> {fmt(ch['HTML tags'])} reviews contain HTML tags that would create a 'br' token, and "
        f"{fmt(ch['Hyphenated words'])} contain hyphenated words (feel-good) whose parts mean something different alone. "
        f"(3) <b>Rare symbols:</b> {fmt(ch['Non-ASCII / emoji characters'])} reviews contain non-ASCII characters or emoji, which whitespace and regex tokenizers treat inconsistently."))

    s.append(para("4. Cleaning and Normalisation Pipeline", H1))
    s.append(para("All steps live in one function, <font name='Courier'>clean_text()</font> in src/clean_text.py, applied identically to every review in this order:"))
    s.append(grid([
        ["#", "Step", "Justification for this data"],
        ["1", "Unicode NFKC normalisation", "Done first so visually identical characters (curly quotes, full-width forms) become one code point before any regex runs. English text, so no script-specific rules are needed."],
        ["2", "HTML entity decoding and tag removal", "Reviews contain <br /> tags; decoding entities first ensures escaped tags are also caught. Tags are markup, not opinion."],
        ["3", "URL and e-mail removal", "Links carry no sentiment and inflate the vocabulary with unique strings."],
        ["4", "Rating normalisation (7/10 to rating7)", "A score is strong sentiment evidence; this keeps it as one token instead of the fragments 7 and 10."],
        ["5", "Case folding (lowercase)", "Merges The/the; shrinks the vocabulary. Trade-off: ALL-CAPS emphasis is lost, accepted because the effect is small relative to vocabulary sparsity."],
        ["6", "Contraction expansion", "Run after lowercasing and before tokenizing so n't becomes 'not' and is kept, instead of leaving junk tokens."],
        ["7", "Regex tokenization [a-z]+\\d*|\\d+", "Drops punctuation and symbols; keeps words and numbers."],
        ["8", f"Stop-word removal (NLTK list, {cl['stopword_count']} words)", "Removes function words. Words that carry sentiment or negation are protected: " + ", ".join(cl["kept_words"]) + "."],
        ["9", "Porter stemming", "See below."],
    ], [W * 0.04, W * 0.27, W * 0.69]))
    s.append(para(
        "<b>Stemming vs lemmatization.</b> Porter stemming was chosen. Reviews are long (50k documents) and Phase 2 uses bag-of-words style features, "
        "so merging inflected forms matters more than producing dictionary words. Stemming is fast, deterministic and needs no POS tagger or model download; "
        "lemmatization without POS tags leaves verbs unchanged (watching stays watching) and with a tagger is much slower. The cost is non-words such as 'movi'. "
        "Examples: " + safe(", ".join(f"{a} -> {b}" for a, b in cl["stem_examples"])) + "."))
    s.append(grid([
        ["Vocabulary", "Size"],
        ["Before cleaning (raw text, case-sensitive \\w+ tokens)", fmt(cl["vocab_raw_case_sensitive"])],
        ["Before cleaning (raw text, lowercased)", fmt(cl["vocab_raw_lowercase"])],
        ["After cleaning", fmt(cl["vocab_cleaned"])],
        ["Reduction", f"{cl['reduction_vs_raw_pct']}% vs case-sensitive raw, {cl['reduction_vs_lowercase_pct']}% vs lowercased raw"],
        ["Rows kept / dropped (empty after cleaning)", f"{fmt(cl['rows_out'])} / {fmt(cl['rows_dropped_empty'])}"],
    ], [W * 0.55, W * 0.45]))
    s.append(Spacer(1, 3))
    s.append(grid([["Before", "After"], [cl["example_before"][:330], cl["example_after"][:330]]], [W * 0.5, W * 0.5]))

    s.append(para("5. Reproducibility", H1))
    s.append(Preformatted(
        "git clone <repo-url> && cd NLP_Assignment_1 && git checkout assignment-1\n"
        "python -m venv .venv && source .venv/bin/activate   (Windows: .venv\\Scripts\\activate)\n"
        "pip install -r requirements.txt\n"
        "python run_all.py        (download, explore, tokenize, clean, build report)\n"
        "Output: data/processed/cleaned_reviews.csv  (columns: cleaned_review, sentiment)", MONO))
    s.append(para(f"Exact tagged commit: {safe(team['repository_url'])} at tag <b>{safe(team['tag'])}</b>. "
                  "The seed is fixed (42) so sampling and results are identical on every run."))

    s.append(para("6. Execution Evidence (console output of each script)", H1))
    for title, name in [("Exploration", "explore"), ("Tokenization", "tokenization"), ("Cleaning", "clean_text")]:
        s.append(para(title, H2))
        s.append(log_block(name))

    s.append(para("7. Contribution Statement", H1))
    s.append(grid([["Member", "Roll number", "Contribution"]]
                  + [[m["name"], m["roll_number"], m["contribution"]] for m in team["members"]],
                  [W * 0.25, W * 0.15, W * 0.6]))

    doc = SimpleDocTemplate(str(REPORT_FILE), pagesize=A4,
                            leftMargin=1.6 * cm, rightMargin=1.6 * cm,
                            topMargin=1.4 * cm, bottomMargin=1.3 * cm,
                            title="Assignment 1 - Phase 1 Report")
    doc.build(s)
    print(f"Report written to {REPORT_FILE}")
    if "TODO" in json.dumps(team):
        print("Warning: team.json still contains TODO placeholders - fill them in and rebuild.")


if __name__ == "__main__":
    build()
