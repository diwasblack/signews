import logging
import statistics

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.model_selection import train_test_split

from crits.classifier import CriticalTextClassifier
from crits.dataset import CriticalTextDataset


def train_test_svm():
    # Load the critical text dataset
    criticaltext_dataset = CriticalTextDataset()
    tweets, labels = criticaltext_dataset.load_dataset()

    x_train, x_test, y_train, y_test = train_test_split(
        tweets,
        labels,
        test_size=0.2
    )

    # Initialize test classifier
    classifier = CriticalTextClassifier(vectorizer="tfidf")

    logging.info("Training a NN")

    parameters_values = [
        {
            "hidden_layer_sizes": 500,
            "alpha": 0.01
        },
        {
            "hidden_layer_sizes": 500,
            "alpha": 0.1
        },
        {
            "hidden_layer_sizes": 250,
            "alpha": 0.1
        }
    ]

    best_parameters = parameters_values[0]
    best_score = 0

    for parameters in parameters_values:
        # Construct the classifier to use
        clf = MLPClassifier(**parameters)

        score = classifier.validate_model(x_train, y_train, classifier=clf)

        if(score > best_score):
            best_score = score
            best_parameters = parameters

    logging.info("Best parameters {}".format(best_parameters))

    # Construct the classifier with best C value
    clf = MLPClassifier(**best_parameters)
    classifier.fit(x_train, y_train, classifier=clf)

    y_pred = [classifier.predict(tweet) for tweet in x_test]

    accuracy = accuracy_score(y_test, y_pred)
    performance_metrics = precision_recall_fscore_support(
        y_test, y_pred)

    precision = statistics.mean(performance_metrics[0])
    recall = statistics.mean(performance_metrics[1])
    fscore = statistics.mean(performance_metrics[2])

    logging.info(
        f"Accuracy:{accuracy}, Precision:{precision}, Recall:{recall}, Fscore:{fscore}"
    )

    fn_file_path = open("false_negative.txt", "w")
    fp_file_path = open("false_positive.txt", "w")

    for index, value in enumerate(y_test):
        if value == 1 and y_pred[index] == 0:
            fn_file_path.write("{}\n$$end$$\n".format(x_test[index]))
        if value == 0 and y_pred[index] == 1:
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

    train_test_svm()
