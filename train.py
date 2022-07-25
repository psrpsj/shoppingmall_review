from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    HfArgumentParser,
    TrainingArguments
)

def main():
    parser = HfArgumentParser(TrainingArguments)
    training_args = parser.parse_args_into_dataclasses()
    