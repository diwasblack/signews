import nltk


from .stemmer import Stemmer


class TextTokenizer():

    def __init__(self, filter_stopwords=False):
        self.filter_stopwords = filter_stopwords
        self.tokenizer = nltk.tokenize.RegexpTokenizer(r"\w+")

        if(self.filter_stopwords):
            self.filter_stopwords
            self.stopwords = set(nltk.corpus.stopwords.words("english"))

    def tokenize_text(self, text):
        text_data = text.lower()
        tokens = self.tokenizer.tokenize(text_data)

        if(self.filter_stopwords):
            tokens = [
                token for token in tokens if token not in self.stopwords]

        return tokens


class StemTokenizer():
    """
    Class to tokenize the given text and return the stems
    """

    def __init__(self, filter_stopwords=False):
        self.stemmer = Stemmer()
        self.tokenizer = TextTokenizer(filter_stopwords)

    def get_tokens(self, text):
        tokens = self.tokenizer.tokenize_text(text)
        stems = [self.stemmer.stem(token) for token in tokens]
        return stems
