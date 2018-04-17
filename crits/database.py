import os

import peewee
from peewee import TextField, BooleanField, CharField


database_path = os.path.join(
    os.path.dirname(__file__),
    "database.db"
)
db = peewee.SqliteDatabase(database_path)


class Tweet(peewee.Model):
    tweet_id = CharField(unique=True)
    body = TextField()
    is_critical = BooleanField(default=False)

    class Meta:
        database = db


if __name__ == "__main__":
    db.connect()
    db.create_tables([Tweet])
