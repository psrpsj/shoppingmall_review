from dataclasses import dataclass, field
from transformers import TrainingArguments


@dataclass
class ModelArguments:
    """
    Arguments containing information related to model
    """

    model_name: str = field(
        default="klue/bert-base",
        metadata={"help: model name to train from huggingface"},
    )


@dataclass
class TrainingArguments(TrainingArguments):
    output_dir: str = field(default="./output")
    num_train_epochs: int = field(
        default=10, metadata={"help": "number of epoch to run"}
    )
    per_device_train_batch_size: int = field(default=32)
    per_device_eval_batch_size: int = field(default=32)
    overwrite_output_dir: bool = field(default=True)
    save_total_limit: int = field(default=5)
    load_best_model_at_end: bool = field(default=True)
    metric_for_best_model: str = field(default="accuracy")
