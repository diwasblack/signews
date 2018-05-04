import logging
import random

from sklearn.metrics import precision_recall_fscore_support

from crits.classifier import CriticalTextDetector, CriticalTextClassifier
from crits.database import Tweet


def detect_critical_tweets():
    critical_text_detector = CriticalTextDetector()

    # Obtain all tweets
    tweets = Tweet.select()

    for tweet in tweets:
        if(critical_text_detector.detect(tweet.body)):
            tweet.is_critical = 1
        else:
            tweet.is_critical = 0

        tweet.save()


def train_classifier():
    logging.basicConfig(
        format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Obtain the training data
    tweet_objects = Tweet.select(Tweet.body).where(Tweet.is_critical)
    tweets = [tweet.body for tweet in tweet_objects]

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
    logger.info(precision_recall_fscore_support(
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
    train_classifier()
