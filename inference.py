import pandas as pd
import torch

from transformers import AutoTokenizer


def main(args):
    model_name = "klue/bert-base"
    data_path = "./dataset/test.csv"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    dataset = pd.read_csv(data_path)
    dataset["label"] = [100] * len(dataset)
