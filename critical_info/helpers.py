import json

from sklearn.feature_extraction.text import TfidfVectorizer

from stemmer import Stemmer
from classifier import CriticalTextDetector
from database import Tweet
from twitter import TwitterAPI


def obtain_tweets():
    twitter_api = TwitterAPI()

    screen_names = [
        "nypdnews",
        "metpoliceuk",
        "VictoriaPolice",
        "SeattlePD",
        "NYPDCT",
    ]

    for screen_name in screen_names:
        max_id = None
        for i in range(5):
            response, content = twitter_api.get_user_timeline(
                screen_name, max_id=max_id)
            parsed_json_content = json.loads(content)
            for tweet_data in parsed_json_content:
                tweet_text = tweet_data["full_text"]
                tweet_id = tweet_data["id_str"]

                try:
                    tweet = Tweet(tweet_id=tweet_id, body=tweet_text)
                    tweet.save()
                except:
                    pass

            max_id = tweet_data["id"]


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


def process_words():
    stemmer = Stemmer()

    with open("critical_words.json", "r") as file:
        words = json.load(file)

    stemmed_words = list(set([stemmer.stem(word) for word in words]))

    with open("stemmed_words.json", "w") as file:
        json.dump(stemmed_words, file)


def rank_words():
    # Get all tweets
    tweets = Tweet.select()
    tweet_body = (tweet.body for tweet in tweets)

    vectorizer = TfidfVectorizer()
    vector_representation = vectorizer.fit(tweet_body)

    words = vector_representation.vocabulary_
    idf_values = vector_representation.idf_

    word_idf = [(word, idf_values[words[word]]) for word in words]

    idf_sorted_words = sorted(word_idf, key=lambda x: x[1], reverse=True)
    sorted_words = list(zip(*idf_sorted_words))[0]

    file = open("words.json", "w")
    file.write(json.dumps(sorted_words))
    file.close()
