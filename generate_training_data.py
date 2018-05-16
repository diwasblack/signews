import logging

from crits.classifier import CriticalTextDetector
from crits.database import Tweet
from crits.dataset import CriticalTextDataset


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


def save_critical_tweets():
    """
    Save the critical tweets from database to a pickle file
    """

    # Obtain the critical tweets
    critical_tweets_objects = Tweet.select(Tweet.body).where(Tweet.is_critical)
    critical_tweets = [tweet.body for tweet in critical_tweets_objects]

    dataset = CriticalTextDataset()
    dataset.save_dataset(critical_tweets)


if __name__ == "__main__":
    # detect_critical_tweets()
    save_critical_tweets()
