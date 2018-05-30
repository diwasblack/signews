import logging

from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from crits.classifier import CriticalTextClassifier
from crits.dataset import CriticalTextDataset


def train_classifier():
    # Load the critical text dataset
    criticaltext_dataset = CriticalTextDataset()
    tweets, labels = criticaltext_dataset.load_dataset()

    x_train, x_test, y_train, y_test = train_test_split(
        tweets,
        labels,
        test_size=0.33
    )

    # Initialize classifier
    classifier = CriticalTextClassifier(vectorizer="tfidf")
    classifier.fit(x_train, y_train)

    predicted_class_labels = [
        classifier.predict(tweet) for tweet in x_test]

    logging.info(precision_recall_fscore_support(
        y_test, predicted_class_labels))

    fn_file_path = open("false_negative.txt", "w")
    fp_file_path = open("false_positive.txt", "w")

    for index, value in enumerate(y_test):
        if value == 1 and predicted_class_labels[index] == 0:
            fn_file_path.write("{}\n\n".format(x_test[index]))
        if value == 0 and predicted_class_labels[index] == 1:
            fp_file_path.write("{}\n\n".format(x_test[index]))

    fn_file_path.close()
    fp_file_path.close()


if __name__ == "__main__":
    logging.basicConfig(
        format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO)
    logger = logging.getLogger(__name__)
    train_classifier()
