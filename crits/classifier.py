import os
import json
import logging

from sklearn.svm import OneClassSVM

from .tokenizer import StemTokenizer
from .vectorizer import Doc2Vector, TFIDF


class CriticalTextDetector():
    """
    Class to find the critical text based on keywords
    """

    def __init__(self):
        words_path = os.path.join(os.path.dirname(__file__),
                                  "critical_words.json")
        with open(words_path, "r") as file:
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


class CriticalTextClassifier():
    """
    Classifier for critical text
    """

    def __init__(self, vectorizer="word2vec", nu=0.1):
        # Use Doc2Vector as default vectorizer
        if(vectorizer == "word2vec"):
            logging.info("Loading word2vec model from binary file")
            self.vectorizer = Doc2Vector()
        elif(vectorizer == "tfidf"):
            self.vectorizer = TFIDF()
            self.vectorizer.load_idf_values()

        self.classifier = OneClassSVM(nu=nu)

    def fit(self, training_data):
        training_vectors = self.vectorizer.convert_corpus_to_vectors(
            training_data)

        logging.info("Training the classifier")
        self.classifier.fit(training_vectors)

    def predict(self, text):
        document_vector = self.vectorizer.get_vector(text)

        return self.classifier.predict(document_vector.reshape(1, -1))[0]
