from crits.vectorizer import TFIDF

from crits.database import Tweet


def calculate_idf():
    # Obtain all tweets
    tweet_objects = Tweet.select(Tweet.body)
    tweets = [tweet.body for tweet in tweet_objects]

    tf_idf_vectorizer = TFIDF()
    tf_idf_vectorizer.calculate_idf(tweets)

    # Store the vocabulary in a file
    tf_idf_vectorizer.store_vocabulary()

    word_idf_list = tf_idf_vectorizer.get_words_idf()

    with open("idf.txt", "w") as file:
        for word, idf in word_idf_list:
            file.write("{},{}\n".format(word, idf))


if __name__ == "__main__":
    calculate_idf()
