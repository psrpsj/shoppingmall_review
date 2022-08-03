import numpy as np
import os
import pandas as pd
import torch
import torch.nn.functional as F

from arguments import InferenceArgument
from dataset import CustomDataset
from preprocess import preprocess
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    HfArgumentParser,
)


def main():
    parser = HfArgumentParser(InferenceArgument)
    (inference_args,) = parser.parse_args_into_dataclasses()
    data_path = inference_args.data_path
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained(inference_args.base_model)

    dataset = preprocess(data_path, "test.csv")
    dataset["label"] = [100] * len(dataset)
    test_id = dataset["id"]
    test_dataset = CustomDataset(dataset["reviews"], dataset["label"], tokenizer)
    dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # K-Fold Inference using Soft-Voting
    if inference_args.k_fold:
        pred_prob = []
        for fold_num in range(1, 6):
            print(f"---- START INFERENCE FOLD {fold_num} ----")
            model_path = os.path.join(
                "./output/",
                inference_args.project_name + "_kfold",
                "fold" + str(fold_num),
            )
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
            model.resize_token_embeddings(len(tokenizer))
            model.to(device)
            model.eval()

            output_prob = []

            for data in tqdm(dataloader):
                with torch.no_grad():
                    outputs = model(
                        input_ids=data["input_ids"].to(device),
                        attention_mask=data["attention_mask"].to(device),
                        token_type_ids=data["token_type_ids"].to(device),
                    )
                    logits = outputs[0]
                    prob = F.softmax(logits, dim=-1).detach().cpu().numpy()
                    output_prob.append(prob)
            output_prob = np.concatenate(output_prob, axis=0).tolist()
            pred_prob.append(output_prob)

        pred_prob = np.sum(pred_prob, axis=0) / 5
        pred_answer = np.argmax(pred_prob, axis=-1)

        output = pd.DataFrame({"id": test_id, "target": pred_answer})
        output.to_csv(
            os.path.join(
                "./output", inference_args.project_name + "_kfold", "submission.csv"
            ),
            index=False,
        )

    else:
        model_path = os.path.join("./output/", inference_args.project_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.resize_token_embeddings(len(tokenizer))
        model.to(device)
        model.eval()

        output_prob = []
        output_pred = []

        print("---- START INFERENCE ----")
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
        output = pd.DataFrame({"id": test_id, "target": pred_answer})
        output.to_csv(os.path.join("./output", inference_args.project_name, "submission.csv"), index=False)
    print("---- FINISH ----")


if __name__ == "__main__":
    main()
