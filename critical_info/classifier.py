import json

import nltk

from stemmer import Stemmer


class CriticalTextDetector():
    """
    Class to find the critical text based on keywords
    """

    def __init__(self):
        with open("critical_words.json", "r") as file:
            # Construct a lookup table for keywords
            self.keywords = set(json.load(file))
        self.stemmer = Stemmer()

    def detect(self, text):
        text_data = text.lower()
        tokens = nltk.word_tokenize(text_data)

        count = 0
        for token in tokens:
            stemmed_token = self.stemmer.stem(token)
            if stemmed_token in self.keywords:
                count += 1

        if count >= 3:
            return True
        else:
            return False
