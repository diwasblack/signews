import logging

from crits.classifier import CriticalTextDetector
from crits.database import Tweet


def detect_critical_tweets():
    """
    Label the tweets in the database using the critical word list
    """
    logging.info("Labelling tweets using critical words")
    critical_text_detector = CriticalTextDetector()

    # Obtain all tweets
    tweets = Tweet.select()

    for tweet in tweets:
        if(critical_text_detector.detect(tweet.body)):
            tweet.is_critical = 1
        else:
            tweet.is_critical = 0

        tweet.save()


if __name__ == "__main__":
    detect_critical_tweets()
