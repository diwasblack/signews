import os

import numpy as np
from gensim.models import KeyedVectors

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
