import os
from urllib.parse import urlencode

import oauth2 as oauth

TWITTER_KEY = os.environ["TWITTER_KEY"]
TWITTER_SECRET = os.environ["TWITTER_SECRET"]


class TwitterAPI():
    """
    Top level class to handle all twitter API calls
    """

    def send_request(self, request_url, method="GET"):
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

    def get_user_timeline(self, screen_name, max_id=None):
        url = "https://api.twitter.com/1.1/statuses/user_timeline.json"

        url_parameters = {
            "screen_name": screen_name,
            "count": "200",
            "trim_user": "true",
            "exclude_replies": "true",
            "tweet_mode": "extended",
        }

        if max_id:
            url_parameters["max_id"] = max_id

        url = "{}?{}".format(url, urlencode(url_parameters))

        return self.send_request(url)


def main():
    twitter_api = TwitterAPI()
    response, content = twitter_api.get_user_timeline("nypdnews")
    print(content)


if __name__ == "__main__":
    main()
