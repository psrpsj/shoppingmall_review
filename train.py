import argparse
import pandas as pd
import torch
import wandb

from arguments import TrainingArguments
from dataset import CustomDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    HfArgumentParser,
    Trainer,
    set_seed,
)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    probs = pred.predictions

    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}


def main(args):
    model_name = args.model_name
    data_path = args.data_path
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    parser = HfArgumentParser(TrainingArguments)
    (training_args,) = parser.parse_args_into_dataclasses()

    print(f"Current Model is {model_name}")
    print(f"Current device is {device}")

    set_seed(training_args.seed)
    total_dataset = pd.read_csv(data_path)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name)

    # Non K-Fold Process
    model_config = AutoConfig.from_pretrained(pretrained_model_name_or_path=model_name)
    model_config.num_labels = 6
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, config=model_config
    )
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    model.train()

    wandb.init(
        entity="psrpsj",
        project="shoppingmall",
        name=args.project_name,
        tags=model_name,
    )

    wandb.config.update(training_args)

    train_dataset, valid_dataset = train_test_split(
        total_dataset, test_size=0.2, stratify=total_dataset["target"], random_state=42
    )
    train = CustomDataset(train_dataset, train_dataset["target"].tolist(), tokenizer)
    valid = CustomDataset(valid_dataset, valid_dataset["target"].tolist(), tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=valid,
        compute_metrics=compute_metrics,
    )

    print("---- START TRAINING ----")
    trainer.train()
    model.save_pretrained(training_args.output_dir + args.project_name)
    print("---- FINISH ----")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="klue/bert-base")
    parser.add_argument("--project_name", type=str, default="baseline")
    parser.add_argument("--data_path", type=str, default="./dataset/train.csv")
    parser.add_argument("--k_fold", type=bool, default=False)

    args = parser.parse_args()
    main(args)
