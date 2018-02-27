from urllib.parse import urlencode

from twitter import send_request


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
