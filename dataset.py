from torch.utils.data import Dataset
import torch


class CustomDataset(Dataset):
    def __init__(self, dataset, tokenizer):
        self.dataset = dataset
        self.label = dataset["target"]
        self.tokenized_sentence = tokenizer(
            dataset["reviews"].tolist(),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length = 256,
        )

    def __getitem__(self, idx):
        encoded = {
            key: val[idx].clone().detach()
            for key, val in self.tokenized_sentence.items()
        }
        encoded["label"] = torch.tensor(self.label.iloc[idx])
        return encoded

    def __len__(self):
        return len(self.dataset)
