from critical_info.classifier import CriticalTextDetector, CriticalTextClassifier
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


def train_classifier():

    # Obtain the training data
    tweet_objects = Tweet.select(Tweet.body).where(Tweet.is_critical == 1)
    tweets = [tweet.body for tweet in tweet_objects]

    # Initialize classifier
    classifier = CriticalTextClassifier()
    classifier.fit(tweets)

    class_label = [classifier.predict(tweet) for tweet in tweets]
    training_accuracy = class_label.count(1) / len(class_label)

    print("Accuracy: {}".format(training_accuracy))


if __name__ == "__main__":
    train_classifier()
