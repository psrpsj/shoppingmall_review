import os
import pandas as pd
import torch
import wandb

from arguments import TrainModelArgument, TrainingArguments
from dataset import CustomDataset
from preprocess import preprocess
from sklearn.model_selection import StratifiedKFold, train_test_split
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


def main():
    parser = HfArgumentParser(
        (
            TrainingArguments,
            TrainModelArgument,
        )
    )
    (
        training_args,
        model_args,
    ) = parser.parse_args_into_dataclasses()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    print(f"Current Model is {model_args.model_name}")
    print(f"Current device is {device}")

    total_dataset = preprocess(model_args.data_path, train=True)
    total_review = total_dataset["reviews"]
    total_label = total_dataset["target"]
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_args.model_name
    )
    set_seed(training_args.seed)

    # K-Fold Process
    if model_args.k_fold:
        print("---- STARTING K-FOLD ----")
        fold_num = 1
        k_fold = StratifiedKFold(n_splits=5, shuffle=False)
        for train_index, valid_index in k_fold.split(total_dataset, total_label):
            wandb.init(
                entity="psrpsj",
                project="shoppingmall",
                name=model_args.project_name + "_kfold_" + str(fold_num),
                tags=model_args.model_name,
            )
            wandb.config.update(training_args)

            print(f"---- Fold Number {fold_num} start ----")
            output_dir = os.path.join(
                training_args.output_dir,
                model_args.project_name + "_kfold",
                "fold" + str(fold_num),
            )

            train_review, valid_review = (
                total_review.iloc[train_index],
                total_review.iloc[valid_index],
            )

            train_label, valid_label = (
                total_label.iloc[train_index],
                total_label.iloc[valid_index],
            )

            train = CustomDataset(train_review, train_label, tokenizer)
            valid = CustomDataset(valid_review, valid_label, tokenizer)

            model_config = AutoConfig.from_pretrained(
                pretrained_model_name_or_path=model_args.model_name
            )
            model_config.num_labels = 6
            model = AutoModelForSequenceClassification.from_pretrained(
                model_args.model_name, config=model_config
            )
            model.resize_token_embeddings(len(tokenizer))
            model.to(device)
            model.train()

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train,
                eval_dataset=valid,
                compute_metrics=compute_metrics,
            )
            trainer.train()
            model.save_pretrained(output_dir)
            wandb.finish()
            fold_num += 1

    # Non K-Fold Process
    else:
        model_config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path=model_args.model_name
        )
        model_config.num_labels = 6
        model = AutoModelForSequenceClassification.from_pretrained(
            model_args.model_name, config=model_config
        )
        model.resize_token_embeddings(len(tokenizer))
        model.to(device)
        model.train()

        wandb.init(
            entity="psrpsj",
            project="shoppingmall",
            name=model_args.project_name,
            tags=model_args.model_name,
        )

        wandb.config.update(training_args)

        train_dataset, valid_dataset = train_test_split(
            total_dataset, test_size=0.2, stratify=total_label, random_state=42
        )
        train = CustomDataset(
            train_dataset["reviews"], train_dataset["target"], tokenizer
        )
        valid = CustomDataset(
            valid_dataset["reviews"], valid_dataset["target"], tokenizer
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train,
            eval_dataset=valid,
            compute_metrics=compute_metrics,
        )

        print("---- START TRAINING ----")
        trainer.train()
        model.save_pretrained(training_args.output_dir + model_args.project_name)
        wandb.finish()

    print("---- FINISH ----")


if __name__ == "__main__":
    main()
