import logging
import random

from sklearn.metrics import precision_recall_fscore_support

from crits.classifier import CriticalTextDetector, CriticalTextClassifier
from crits.database import Tweet

logging.basicConfig(
    format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


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
    print(precision_recall_fscore_support(label_vectors, predicted_class_labels))


if __name__ == "__main__":
    train_classifier()
