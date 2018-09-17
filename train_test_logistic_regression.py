import logging

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from crits.classifier import CriticalTextClassifier
from crits.dataset import CriticalTextDataset


def train_test_logistic_regression():
    # Load the critical text dataset
    criticaltext_dataset = CriticalTextDataset()
    tweets, labels = criticaltext_dataset.load_dataset()

    x_train, x_test, y_train, y_test = train_test_split(
        tweets,
        labels,
        test_size=0.33
    )

    # Initialize test classifier
    classifier = CriticalTextClassifier(vectorizer="tfidf")

    C_values = [1, 50, 100, 200, 300, 400]
    best_C = C_values[0]
    best_score = 0
    for c_value in C_values:
        # Construct the classifier to use
        clf = LogisticRegression(C=c_value)

        score = classifier.validate_model(x_train, y_train, classifier=clf)

        if(score > best_score):
            best_score = score
            best_C = c_value

    logging.info("Best C={}".format(best_C))

    # Construct the classifier with best C value
    clf = LogisticRegression(C=best_C)
    classifier.fit(x_train, y_train, classifier=clf)

    predicted_class_labels = [
        classifier.predict(tweet) for tweet in x_test]

    logging.info(precision_recall_fscore_support(
        y_test, predicted_class_labels))

    fn_file_path = open("false_negative.txt", "w")
    fp_file_path = open("false_positive.txt", "w")

    for index, value in enumerate(y_test):
        if value == 1 and predicted_class_labels[index] == 0:
            fn_file_path.write("{}\n$$end$$\n".format(x_test[index]))
        if value == 0 and predicted_class_labels[index] == 1:
            fp_file_path.write("{}\n$$end$$\n".format(x_test[index]))

    fn_file_path.close()
    fp_file_path.close()


if __name__ == "__main__":
    logging.basicConfig(
        format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO)

    logger = logging.getLogger(__name__)

    file_handler = logging.FileHandler("classifier.log")
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    logger.addHandler(console_handler)

    train_test_logistic_regression()