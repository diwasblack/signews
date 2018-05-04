import unittest


from crits.tokenizer import TextTokenizer


class TestTextTokenizer(unittest.TestCase):
    def setUp(self):
        self.tokenizer = TextTokenizer()

    def test_link_strip(self):
        test_string = "hey how are you https://t.co/eln9wukga2 man"
        tokens = self.tokenizer.tokenize_text(test_string)

        self.assertEqual(len(tokens), 5)


if __name__ == "__main__":
    unittest.main()
