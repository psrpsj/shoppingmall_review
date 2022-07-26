import torch
import wandb

from arguments import TrainingArguments

from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)


def main():
    model_name = "klue/bert-base"
    data_path = "./dataset/train.csv"
    device = torch.device("cuda") if torch.cuda.is_avaliable() else torch.device("cpu")
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

    wandb.init(
        entity="psrpsj",
        project="shoppingmall",
        name=model_name,
        tags=model_name,
    )
