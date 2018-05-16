import os
import pickle


class CriticalTextDataset():
    """
    Class to handle the dataset for the critical text
    """

    def __init__(self):
        self.dataset_path = os.path.join(
            os.path.dirname(__file__),
            "dataset.pkl"
        )

    def load_dataset(self):
        with open(self.dataset_path, "rb") as file:
            return pickle.load(file)
