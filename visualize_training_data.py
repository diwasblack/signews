from sklearn.manifold import TSNE

import matplotlib.pyplot as plt

from crits.vectorizer import TFIDF
from crits.database import Tweet


def visualize_data():
    # Obtain the training data
    tweet_objects = Tweet.select(Tweet.body, Tweet.is_critical)
    tweets_and_labels = [(tweet.body, tweet.is_critical)
                         for tweet in tweet_objects][:2000]

    tweets, labels = list(zip(*tweets_and_labels))

    print("Initializing vectorizer")
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
    visualize_data()
