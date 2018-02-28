import json

from tokenizer import StemTokenizer


class CriticalTextDetector():
    """
    Class to find the critical text based on keywords
    """

    def __init__(self):
        with open("critical_words.json", "r") as file:
            # Construct a lookup table for keywords
            self.keywords = set(json.load(file))
        self.tokenizer = StemTokenizer()

    def detect(self, text):
        tokens = self.tokenizer.get_tokens(text)

        # Find out the set intersection
        common_tokens = self.keywords & set(tokens)
        if len(common_tokens) >= 3:
            return True
        else:
            return False
