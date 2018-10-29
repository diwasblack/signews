from signews.vectorizer import TFIDF


def print_vocab_stems():
    vectorizer = TFIDF()
    vectorizer.load_idf_values()

    # Load sample text
    with open("sample_text.txt", "r") as file:
        sample_text = file.read()

    print(vectorizer.get_vocab_word(sample_text))


if __name__ == "__main__":
    print_vocab_stems()
