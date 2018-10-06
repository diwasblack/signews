from nltk.stem.snowball import SnowballStemmer


class Stemmer():
    def __init__(self):
        self.stemmer = SnowballStemmer("english")

    def stem(self, word):
        return self.stemmer.stem(word)
