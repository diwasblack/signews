from critical_info.classifier import CriticalTextDetector
from critical_info.database import Tweet


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


if __name__ == "__main__":
    detect_critical_tweets()
