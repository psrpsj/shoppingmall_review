import pandas as pd
import torch
import wandb

from arguments import TrainingArguments
from dataset import CustomDataset
from sklearn.model.selection import train_test_split
from sklearn.metrics import accuracy_score
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictionas.argmax(-1)
    probs = pred.predictions

    return accuracy_score(labels, preds)


def main():
    model_name = "klue/bert-base"
    data_path = "./dataset/train.csv"
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    parser = HfArgumentParser(TrainingArguments)
    training_args = parser.parse_args_into_dataclasses()

    print(f"Current Model is {model_name}")
    print(f"Current device is {device}")

    set_seed(training_args.seed)

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name)
    model_config = AutoConfig.from_pretrained(pretrained_model_name_or_path=model_name)
    model_config.num_labels = 4
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, config=model_config
    )
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    model.train()

    wandb.init(
        entity="psrpsj",
        project="shoppingmall",
        name=model_name,
        tags=model_name,
    )

    total_dataset = pd.read_csv(data_path)
    train_dataset, valid_dataset = train_test_split(
        total_dataset, test_size=0.2, startify=total_dataset["target"], random_state=42
    )
    train = CustomDataset(train_dataset, tokenizer)
    valid = CustomDataset(valid_dataset, tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=valid,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    model.save_pretrained(f"./result/{model_name}")


if __name__ == "__main__":
    main()
