import os
from urllib.parse import urlencode

import oauth2 as oauth

TWITTER_KEY = os.environ.get("TWITTER_KEY", "")
TWITTER_SECRET = os.environ.get("TWITTER_SECRET", "")


def send_request(request_url, method="GET"):
    # Create your consumer with the proper key/secret.
    consumer = oauth.Consumer(
        key=TWITTER_KEY,
        secret=TWITTER_SECRET
    )

    # Create our client.
    client = oauth.Client(consumer)

    # The OAuth Client request works just like httplib2 for the most part.
    response, content = client.request(request_url, method)

    # Try to decode the content
    content = content.decode()

    return response, content


def main():
    # url = "https://api.twitter.com/1.1/search/tweets.json?q=from%3ANasa%20OR%20%23nasa"
    # url = "https://api.twitter.com/1.1/statuses/show.json?id=962487358028484608"

    # url = "https://stream.twitter.com/1.1/statuses/sample.json"
    # url = "https://stream.twitter.com/1.1/statuses/filter.json?track=twitter"

    # url = "https://api.twitter.com/1.1/trends/place.json?id=1"

    url = "https://api.twitter.com/1.1/statuses/user_timeline.json"

    url_parameters = {
        "screen_name": "nypdnews",
        "count": "3200",
        "trim_user": "true",
        "exclude_replies": "true",
        "tweet_mode": "extended"
    }

    url = "{}?{}".format(url, urlencode(url_parameters))

    response, content = send_request(url, "GET")

    with open("response.json", "w") as file:
        print(content)
        file.write(content)


if __name__ == "__main__":
    main()
