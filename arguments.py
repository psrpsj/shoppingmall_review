from dataclasses import dataclass, field
from transformers import TrainingArguments


@dataclass
class TrainingArguments(TrainingArguments):
    output_dir: str = field(default="./output/")
    num_train_epochs: int = field(
        default=5, metadata={"help": "number of epoch to run"}
    )
    per_device_train_batch_size: int = field(default=8)
    per_device_eval_batch_size: int = field(default=8)
    overwrite_output_dir: bool = field(default=True)
    save_total_limit: int = field(default=5)
    load_best_model_at_end: bool = field(default=True)
    evaluation_strategy: str = field(default = "steps")
    save_strategy: str = field(default="steps")
    metric_for_best_model: str = field(default="accuracy")
    full_determinism: bool = field(default=False)
