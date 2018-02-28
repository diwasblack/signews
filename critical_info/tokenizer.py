import nltk

from stemmer import Stemmer


def tokenize_text(text):
    text_data = text.lower()
    tokens = nltk.word_tokenize(text_data)
    return tokens


class StemTokenizer():
    """
    Class to tokenize the given text and return the stems
    """

    def __init__(self):
        self.stemmer = Stemmer()

    def get_tokens(self, text):
        tokens = tokenize_text(text)
        stems = [self.stemmer.stem(token) for token in tokens]
        return stems
