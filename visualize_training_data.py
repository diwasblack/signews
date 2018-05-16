import logging

from sklearn.manifold import TSNE

import matplotlib.pyplot as plt

from crits.vectorizer import TFIDF
from crits.dataset import CriticalTextDataset


def visualize_data():
    criticaltext_dataset = CriticalTextDataset()
    tweets, labels = criticaltext_dataset.load_dataset()

    logging.info("Initializing vectorizer")
    vectorizer = TFIDF()
    vectorizer.load_idf_values()
    tsne_object = TSNE()

    training_vectors = vectorizer.convert_corpus_to_vectors(
        tweets)

    reduced_vectors = tsne_object.fit_transform(training_vectors)

    critical_tweet_vectors = []
    non_critical_tweet_vectors = []

    for index, vector in enumerate(reduced_vectors):
        if labels[index]:
            critical_tweet_vectors.append(vector)
        else:
            non_critical_tweet_vectors.append(vector)

    plt.scatter(*list(zip(*critical_tweet_vectors)), c="r")
    plt.scatter(*list(zip(*non_critical_tweet_vectors)), c="b")
    plt.savefig("data_visualization.png")


if __name__ == "__main__":
    logging.basicConfig(
        format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO)
    logger = logging.getLogger(__name__)
    visualize_data()
