from datasets import load_dataset


def load_data(path):
    dataset = load_dataset("csv", data_files=path)

    review = ["[CLS] " + str(r) + " [SEP]" for r in dataset["reviews"]]
