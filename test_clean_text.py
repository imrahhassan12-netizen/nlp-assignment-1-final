import unittest

from src.clean_text import STOPWORDS, clean_text


class CleanTextTests(unittest.TestCase):
    def test_html_removed(self):
        self.assertNotIn("br", clean_text("Great film.<br /><br />Loved it").split())

    def test_negation_kept(self):
        tokens = clean_text("I didn't like it, it wasn't good").split()
        self.assertEqual(tokens.count("not"), 2)

    def test_contraction_has_no_stray_tokens(self):
        tokens = clean_text("you'll be hooked, they're great").split()
        self.assertNotIn("ll", tokens)
        self.assertNotIn("t", tokens)

    def test_url_and_rating(self):
        tokens = clean_text("Visit www.example.com. Score 8/10").split()
        self.assertIn("rating8", tokens)
        self.assertFalse(any("example" in t for t in tokens))

    def test_lowercase_and_stem(self):
        self.assertEqual(clean_text("WATCHING Movies"), "watch movi")

    def test_sentiment_words_protected(self):
        for word in ("not", "no", "never", "very", "too"):
            self.assertNotIn(word, STOPWORDS)

    def test_deterministic(self):
        text = "It's <b>GREAT</b>!!! 9/10"
        self.assertEqual(clean_text(text), clean_text(text))


if __name__ == "__main__":
    unittest.main()
