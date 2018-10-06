import logging

from signews.classifier import SignificantTextDetector
from signews.database import Tweet


def detect_significant_tweets():
    """
    Label the tweets in the database using the critical word list
    """
    logging.info("Labelling tweets using critical words")
    critical_text_detector = SignificantTextDetector()

    # Obtain all tweets
    tweets = Tweet.select()

    for tweet in tweets:
        if(critical_text_detector.detect(tweet.body)):
            tweet.is_critical = 1
        else:
            tweet.is_critical = 0

        tweet.save()


if __name__ == "__main__":
    detect_significant_tweets()
