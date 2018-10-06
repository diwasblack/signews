import logging

from sklearn.manifold import TSNE

import matplotlib.pyplot as plt

from signews.vectorizer import TFIDF
from signews.dataset import SignificantTextDataset


def visualize_data():
    dataset = SignificantTextDataset()
    tweets, labels = dataset.load_dataset()

    logging.info("Initializing vectorizer")
    vectorizer = TFIDF()
    vectorizer.load_idf_values()
    tsne_object = TSNE()

    logging.info("Converting text to vectors")
    training_vectors = vectorizer.convert_corpus_to_vectors(
        tweets)

    logging.info("Applying t-SNE")
    reduced_vectors = tsne_object.fit_transform(training_vectors)

    critical_tweet_vectors = []
    non_critical_tweet_vectors = []

    for index, vector in enumerate(reduced_vectors):
        if labels[index]:
            critical_tweet_vectors.append(vector)
        else:
            non_critical_tweet_vectors.append(vector)

    fig, ax = plt.subplots()

    ax.scatter(*list(zip(*critical_tweet_vectors)), c="r", label="Significant")
    ax.scatter(*list(zip(*non_critical_tweet_vectors)), c="b", label="Non-significant")

    ax.legend()
    plt.savefig("data_visualization.png")


if __name__ == "__main__":
    logging.basicConfig(
        format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO)
    logger = logging.getLogger(__name__)
    visualize_data()
