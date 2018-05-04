import os
import pickle
import json

import numpy as np
from gensim.models import KeyedVectors
from sklearn.feature_extraction.text import TfidfVectorizer

from .tokenizer import TextTokenizer, StemTokenizer


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
        self.tokenizer = TextTokenizer(filter_words=True)

        self.vector_length = self.word2vec_model.wv.vectors.shape[1]

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
                # NOTE a vector of zeros may not be the best choice
                word_vector = np.zeros(self.vector_length)

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
        self.tokenizer = StemTokenizer(filter_words=True)

        file_path = os.path.dirname(__file__)

        self.tf_idf_model_path = os.path.join(file_path, "tf_idf.pkl")
        self.vocabulary_file_path = os.path.join(file_path, "vocabulary.json")

        self.tf_idf = None

        self.max_features = 1000

    def load_idf_values(self):
        if(not(os.path.exists(self.tf_idf_model_path))):
            raise Exception("IDF values file not found")
        with open(self.tf_idf_model_path, "rb") as file:
            self.tf_idf = pickle.load(file)

    def calculate_idf(self, corpus, use_fixed_vocab=False):
        """
        Calculate and store the IDF vectors
        """

        if(use_fixed_vocab):
            with open(self.vocabulary_file_path, "r") as vocab_file:
                self.tf_idf = TfidfVectorizer(
                    tokenizer=self.tokenizer.tokenize_text,
                    vocabulary=json.load(vocab_file)
                )
        else:
            self.tf_idf = TfidfVectorizer(
                tokenizer=self.tokenizer.tokenize_text,
                max_features=self.max_features
            )

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

    def get_words_idf(self):
        """
        Return a sorted list of words and their respective IDF values
        """

        words = self.tf_idf.vocabulary_
        idfs = self.tf_idf.idf_

        word_idf_list = [(k, idfs[v]) for k, v in words.items()]
        return sorted(word_idf_list, key=lambda x: x[1], reverse=True)

    def store_vocabulary(self):
        """
        Store the words list in a json file
        """

        words = list(self.tf_idf.vocabulary_.keys())

        with open(self.vocabulary_file_path, "w") as file:
            json.dump(words, file)
