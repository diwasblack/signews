import os

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
