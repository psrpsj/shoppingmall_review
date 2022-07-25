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
    parser = HfArgumentParser(TrainingArguments)
    training_args = parser.parse_args_into_dataclasses()

    print(f"Current Model is {model_name}")

    set_seed(training_args.seed)

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name)
    model_config = AutoConfig.from_pretrained(pretrained_model_name_or_path=model_name)
    model_config.num_labels = 4
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, config=model_config
    )
    model.train()
