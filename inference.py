import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from dataset import CustomDataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataLoader


def main():
    model_path = "./output/klue/bert-base"
    data_path = "./dataset/test.csv"
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification(model_path)
    model.resize_token_embedding(len(tokenizer))
    model.to(device)
    model.eval()

    dataset = pd.read_csv(data_path)
    dataset["label"] = [100] * len(dataset)
    test_id = dataset["id"]
    test_dataset = CustomDataset(dataset, dataset["label"], tokenizer)
    dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    output_prob = []
    output_pred = []
    for data in tqdm(dataloader):
        output = model(
            input_ids=data["input_ids"].to(device),
            attention_mask=data["attention_mask"].to(device),
            token_type_ids=data["token_type_ids"].to(device),
        )

        logit = output[0]
        prob = F.softmax(logit, dim=-1).detach().cpu().numpy()
        logit = logit.detach().cpu().numpy()
        result = np.argmax(logit, axis=-1)
        output_pred.append(result)
        output_prob.append(prob)

    pred_answer = np.concatenate(output_pred).tolist()
    output_prob = np.concatenate(output_prob, axis=0).tolist()
    output = np.DataFrame({"id": test_id, "target": pred_answer})
    output.to_csv("./output/submission.csv")
    print("---- FINISH ----")


if __name__ == "__main__":
    main()
