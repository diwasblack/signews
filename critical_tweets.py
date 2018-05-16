import logging
import random

from sklearn.metrics import precision_recall_fscore_support

from crits.classifier import CriticalTextClassifier
from crits.database import Tweet
from crits.dataset import CriticalTextDataset


def train_classifier():
    # Obtain the training data
    criticaltext_dataset = CriticalTextDataset()
    tweets = criticaltext_dataset.load_dataset()

    # Initialize classifier
    classifier = CriticalTextClassifier(vectorizer="tfidf")
    classifier.fit(tweets)

    test_tweets_objects = Tweet.select(Tweet.body, Tweet.is_critical)
    random_tweets = random.sample([(tweet.body, tweet.is_critical)
                                   for tweet in test_tweets_objects], 2000)

    test_tweets, labels = zip(*random_tweets)
    label_vectors = [1 if x else -1 for x in labels]

    predicted_class_labels = [
        classifier.predict(tweet) for tweet in test_tweets]
    logging.info(precision_recall_fscore_support(
        label_vectors, predicted_class_labels))

    fn_file_path = open("false_negative.txt", "w")
    fp_file_path = open("false_positive.txt", "w")

    for index, value in enumerate(label_vectors):
        if value == 1 and predicted_class_labels[index] == -1:
            fn_file_path.write("{}\n\n".format(random_tweets[index][0]))
        if value == -1 and predicted_class_labels[index] == 1:
            fp_file_path.write("{}\n\n".format(random_tweets[index][0]))

    fn_file_path.close()
    fp_file_path.close()


if __name__ == "__main__":
    logging.basicConfig(
        format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO)
    logger = logging.getLogger(__name__)
    train_classifier()
