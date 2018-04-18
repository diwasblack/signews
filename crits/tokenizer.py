import string

import nltk

from .stemmer import Stemmer


class TextTokenizer():
    def __init__(self, filter_stopwords=False):
        self.character_filter = str.maketrans("", "", string.punctuation)
        self.tokenizer = nltk.tokenize.RegexpTokenizer(r"[a-zA-Z]+")

        self.filter_stopwords = filter_stopwords

        if(self.filter_stopwords):
            self.stopwords = set(nltk.corpus.stopwords.words("english"))

    def tokenize_text(self, text):
        text_data = text.lower()
        text_data = text_data.translate(self.character_filter)
        tokens = self.tokenizer.tokenize(text_data)

        if(self.filter_stopwords):
            tokens = [
                token for token in tokens if token not in self.stopwords]

        return tokens


class StemTokenizer(TextTokenizer):
    """
    Class to tokenize the given text and return the stems
    """

    def __init__(self, filter_stopwords=False):
        super().__init__(filter_stopwords=filter_stopwords)
        self.stemmer = Stemmer()

    def tokenize_text(self, text):
        tokens = super().tokenize_text(text)
        stems = [self.stemmer.stem(token) for token in tokens]
        return stems
