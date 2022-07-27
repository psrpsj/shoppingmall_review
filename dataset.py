import pandas as pd

from datasets import load_dataset


def create_token_data(path):
    dataset = pd.read_csv(path)
    review = ["[CLS] " + str(r) + " [SEP]" for r in dataset["reviews"]]
    dataset["reviews"] = review
    dataset.to_csv("./dataset/train_w_token.csv")
