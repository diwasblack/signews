import peewee
from peewee import TextField, BooleanField


db = peewee.SqliteDatabase("database.db")


class Tweet(peewee.Model):
    tweet_id = TextField()
    body = TextField()
    is_critical = BooleanField()

    class Meta:
        database = db


if __name__ == "__main__":
    db.connect()
    db.create_tables([Tweet])
