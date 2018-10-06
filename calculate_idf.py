from signews.vectorizer import TFIDF
from signews.database import Tweet


def calculate_idf():
    # Obtain all tweets
    tweet_objects = Tweet.select(Tweet.body)
    tweets = [tweet.body for tweet in tweet_objects]

    tf_idf_vectorizer = TFIDF()
    tf_idf_vectorizer.calculate_idf(tweets)

    # Store the word and it's IDF value in a file
    tf_idf_vectorizer.save_word_idf()


if __name__ == "__main__":
    calculate_idf()
