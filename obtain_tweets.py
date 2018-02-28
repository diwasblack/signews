import json

from critical_info.twitter import TwitterAPI
from critical_info.database import Tweet


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


if __name__ == "__main__":
    obtain_tweets()
