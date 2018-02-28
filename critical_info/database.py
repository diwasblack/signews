import peewee
from peewee import TextField, BooleanField, CharField


db = peewee.SqliteDatabase("database.db")


class Tweet(peewee.Model):
    tweet_id = CharField(unique=True)
    body = TextField()
    is_critical = BooleanField(default=False)

    class Meta:
        database = db


if __name__ == "__main__":
    db.connect()
    db.create_tables([Tweet])
