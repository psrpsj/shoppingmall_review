from torch.utils.data import Dataset
import torch


class CustomDataset(Dataset):
    def __init__(self, dataset, tokenizer):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.tokenized_sentence = self.tokenizer(
            dataset["reviews"].tolist(),
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )

    def __getitem__(self, idx):
        encoded = {
            key: val[idx].clone().detach()
            for key, val in self.tokenized_sentence.items()
        }
        encoded["label"] = torch.tensor(self.dataset["target"][idx])
        return encoded

    def __len__(self):
        return len(self.dataset)
