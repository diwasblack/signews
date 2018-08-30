import os
import json
import logging
import pickle
import statistics

from sklearn.svm import SVC
from sklearn.model_selection import cross_val_predict, cross_validate
from sklearn.metrics import precision_recall_fscore_support

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
        tokens = self.tokenizer.tokenize_text(text)

        # Find out the set intersection
        common_tokens = self.keywords & set(tokens)
        if len(common_tokens) >= 2:
            return True
        else:
            return False


class CriticalTextClassifier():
    """
    Classifier for critical text

    Labels critical text as 1 and non critical news as -1
    """

    def __init__(self, vectorizer="word2vec"):
        self.model_path = os.path.join(
            os.path.dirname(__file__), "trained_classifier.pkl")

        # Use Doc2Vector as default vectorizer
        if(vectorizer == "word2vec"):
            logging.info("Loading word2vec model from binary file")
            self.vectorizer = Doc2Vector()
        elif(vectorizer == "tfidf"):
            self.vectorizer = TFIDF()
            self.vectorizer.load_idf_values()

        self.classifier = None

    def fit(self, training_data, labels, C=100):
        training_vectors = self.vectorizer.convert_corpus_to_vectors(
            training_data)

        logging.info("Training the classifier")
        self.classifier = SVC(C=C)
        self.classifier.fit(training_vectors, labels)

    def predict(self, text):
        document_vector = self.vectorizer.get_vector(text)

        return self.classifier.predict(document_vector.reshape(1, -1))[0]

    def validate_model(self, x_train, y_train, C, cv=5):
        logging.info("Converting validation text to vectors")
        x_train_vectors = self.vectorizer.convert_corpus_to_vectors(
            x_train)

        logging.info("Performing k fold cross validation")
        classifier = SVC(C=C)
        cv_results = cross_validate(
            classifier, x_train_vectors, y_train, cv=cv, n_jobs=-1,
            scoring=("precision", "recall", "f1"))

        f1_scores = cv_results["test_f1"]

        average_score = statistics.mean(f1_scores)

        logging.info("Average F score={}".format(average_score))

        return average_score

    def save_model(self):
        """
        Save the trained model to a file
        """

        with open(self.model_path, "wb") as file:
            pickle.dump((self.vectorizer, self.classifier), file)

    def load_model(self):
        """
        Load a pre trained model from a file
        """

        with open(self.model_path, "rb") as file:
            self.vectorizer, self.classifier = pickle.load(file)
