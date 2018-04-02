import os
import pickle

import numpy as np
from gensim.models import KeyedVectors
from sklearn.feature_extraction.text import TfidfVectorizer

from .tokenizer import TextTokenizer


class Doc2Vector():
    """
    Vectorize a text using a word2vec model
    """

    def __init__(self):
        word_vectors_file = os.path.join(os.path.dirname(__file__),
                                         "GoogleNews-vectors-negative300.bin")
        self.word2vec_model = KeyedVectors.load_word2vec_format(
            word_vectors_file, binary=True)

        # Initialize the tokenizer
        self.tokenizer = TextTokenizer(filter_stopwords=True)

    def get_vector(self, text):
        """
        Return the vector representation for given text
        """

        tokens = self.tokenizer.tokenize_text(text)

        word_vectors = []

        for token in tokens:
            try:
                word_vector = self.word2vec_model.wv[token]
            except:
                continue

            word_vectors.append(word_vector)

        word_vectors = np.array(word_vectors)

        # NOTE explore other ways to combine word vectors
        return np.average(word_vectors, axis=0)

    def convert_corpus_to_vectors(self, documents):
        document_vectors = [self.get_vector(doc) for doc in documents]
        return np.array(document_vectors)


class TFIDF():
    """
    Vectorize a given text using TF-IDF
    """

    def __init__(self):
        # Initialize the tokenizer
        self.tokenizer = TextTokenizer()

        self.tf_idf_model_path = os.path.join(os.path.dirname(__file__),
                                              "tf_idf.pkl")

        if(os.path.exists(self.tf_idf_model_path)):
            with open(self.tf_idf_model_path, "rb") as file:
                self.tf_idf = pickle.load(file)
        else:
            self.tf_idf = TfidfVectorizer(max_features=1000)

    def calculate_idf(self, corpus):
        """
        Calculate and store the IDF vectors
        """

        self.tf_idf.fit(corpus)

        with open(self.tf_idf_model_path, "wb") as file:
            pickle.dump(self.tf_idf, file)

    def get_vector(self, text):
        """
        Return the TF-IDF vector representation of given text
        """
        sparse_matrix = self.tf_idf.transform([text])
        return sparse_matrix.toarray()

    def convert_corpus_to_vectors(self, documents):
        training_sparse_matrix = self.tf_idf.transform(documents)
        return training_sparse_matrix.toarray()
