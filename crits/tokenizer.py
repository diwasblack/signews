import re
import os
import string
import json

import nltk

from .stemmer import Stemmer


class TextTokenizer():
    def __init__(self, filter_stopwords=False):
        self.character_filter = str.maketrans("", "", string.punctuation)
        self.tokenizer = nltk.tokenize.RegexpTokenizer(r"[a-zA-Z]+")

        # Regex for removing urls
        self.url_regex = re.compile('https?:\/\/\S*')

        self.filter_stopwords = filter_stopwords

        if(self.filter_stopwords):
            self.stop_words = set(nltk.corpus.stopwords.words("english"))

    def tokenize_text(self, text):
        text_data = text.lower()
        text_data = re.sub(self.url_regex, "", text_data)

        text_data = text_data.translate(self.character_filter)
        tokens = self.tokenizer.tokenize(text_data)

        if(self.filter_stopwords):
            tokens = [
                token for token in tokens if token not in self.stop_words]

        return tokens


class StemTokenizer(TextTokenizer):
    """
    Class to tokenize the given text and return the stems
    """

    def __init__(self, filter_words=False):
        super().__init__(filter_stopwords=filter_words)
        self.stemmer = Stemmer()

        stem_path = os.path.join(
            os.path.dirname(__file__),
            "filter_words.json"
        )

        with open(stem_path, "r") as file:
            self.filter_stems = json.load(file)

    def tokenize_text(self, text):
        tokens = super().tokenize_text(text)
        stems = [self.stemmer.stem(token) for token in tokens]
        stems = [stem for stem in stems if stem not in self.filter_stems]
        return stems
