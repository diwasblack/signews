from signews.classifier import SignificantTextClassifier


def test_classifier():
    classifier = SignificantTextClassifier(vectorizer="tfidf")
    classifier.load_model()

    # Load sample text
    with open("sample_text.txt", "r") as file:
        sample_text = file.read()

    print(classifier.predict(sample_text))


if __name__ == "__main__":
    test_classifier()
