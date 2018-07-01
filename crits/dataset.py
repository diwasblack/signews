import os
import json


class CriticalTextDataset():
    """
    Class to handle the dataset for the critical text
    """

    def __init__(self):
        base_path = os.path.dirname(__file__)
        self.dataset_path = os.path.join(base_path, "dataset.json")

        self.critical_text_path = os.path.join(
            base_path, "dataset_critical.txt")
        self.non_critical_text_path = os.path.join(
            base_path, "dataset_non_critical.txt")

        self.text_seperator = "\n$$end$$\n"

    def load_dataset(self):
        with open(self.dataset_path, "r") as file:
            return json.load(file)

    def dump_data(self):
        """
        Dump data to txt files
        """

        with open(self.dataset_path, "r") as file:
            texts, labels = json.load(file)
            critical_text_file = open(self.critical_text_path, "w")
            non_critical_text_file = open(self.non_critical_text_path, "w")

            for text, label in zip(texts, labels):
                text_with_suffix = "{}{}".format(text, self.text_seperator)
                if(label == 1):
                    critical_text_file.write(text_with_suffix)
                else:
                    non_critical_text_file.write(text_with_suffix)

    def save_data(self):
        """
        Load the data from the txt files
        """

        with open(self.dataset_path, "w") as file:

            critical_text_file = open(self.critical_text_path, "r")
            content = critical_text_file.read()
            critical_texts = content.split(self.text_seperator)
            critical_text_file.close()

            non_critical_text_file = open(self.non_critical_text_path, "r")
            content = non_critical_text_file.read()
            non_critical_texts = content.split(self.text_seperator)
            non_critical_text_file.close()

            labels = [1] * len(critical_texts) + [0] * len(non_critical_texts)
            dataset = critical_texts + non_critical_texts

            json.dump((dataset, labels), file)
