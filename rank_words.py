import pickle
import json

from sklearn.feature_extraction.text import TfidfVectorizer

from database import Tweet


def main():
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


if __name__ == "__main__":
    main()
