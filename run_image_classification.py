import os
os.environ["USE_TF"] = "0"
import logging
import sys
from dataclasses import dataclass, field
from typing import Optional
import re
import evaluate
import numpy as np
import torch
import mlflow
from datasets import load_dataset
from PIL import Image
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Lambda,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)

import transformers
from transformers import (
    MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING,
    AutoConfig,
    AutoImageProcessor,
    AutoModelForImageClassification,
    HfArgumentParser,
    TimmWrapperImageProcessor,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

require_version("datasets>=2.14.0", "To fix: pip install -r examples/pytorch/image-classification/requirements.txt")

MODEL_CONFIG_CLASSES = list(MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def pil_loader(path: str):
    with open(path, "rb") as f:
        im = Image.open(f)
        return im.convert("RGB")


@dataclass
class DataTrainingArguments:
    dataset_name: Optional[str] = field(default=None)
    dataset_config_name: Optional[str] = field(default=None)
    train_dir: Optional[str] = field(default=None)
    validation_dir: Optional[str] = field(default=None)
    train_val_split: Optional[float] = field(default=0.15)
    max_train_samples: Optional[int] = field(default=None)
    max_eval_samples: Optional[int] = field(default=None)
    image_column_name: str = field(default="image")
    label_column_name: str = field(default="label")

    def __post_init__(self):
        if self.dataset_name is None and (self.train_dir is None and self.validation_dir is None):
            raise ValueError("Must specify dataset_name or train/validation_dir")


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="google/vit-base-patch16-224-in21k")
    model_type: Optional[str] = field(default=None)
    config_name: Optional[str] = field(default=None)
    cache_dir: Optional[str] = field(default=None)
    model_revision: str = field(default="main")
    image_processor_name: str = field(default=None)
    token: str = field(default=None)
    trust_remote_code: bool = field(default=False)
    ignore_mismatched_sizes: bool = field(default=True)


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    send_example_telemetry("run_image_classification", model_args, data_args)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    set_seed(training_args.seed)

    if data_args.dataset_name is not None:
        dataset = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
        )
    else:
        data_files = {}
        if data_args.train_dir is not None:
            data_files["train"] = os.path.join(data_args.train_dir, "**")
        if data_args.validation_dir is not None:
            data_files["validation"] = os.path.join(data_args.validation_dir, "**")
        dataset = load_dataset("imagefolder", data_files=data_files, cache_dir=model_args.cache_dir)

    labels = dataset["train"].features[data_args.label_column_name].names
    label2id = {label: str(i) for i, label in enumerate(labels)}
    id2label = {str(i): label for i, label in enumerate(labels)}

    metric = evaluate.load("accuracy", cache_dir=model_args.cache_dir)

    def compute_metrics(p):
        return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)

    config = AutoConfig.from_pretrained(
        model_args.config_name or model_args.model_name_or_path,
        num_labels=len(labels),
        label2id=label2id,
        id2label=id2label,
        finetuning_task="image-classification",
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    model = AutoModelForImageClassification.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
        ignore_mismatched_sizes=True,
    )

    image_processor = AutoImageProcessor.from_pretrained(
        model_args.image_processor_name or model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )

#    size = image_processor.size.get("shortest_edge") or (image_processor.size["height"], image_processor.size["width"])
    size = (384, 384)
    normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std) if hasattr(image_processor, "image_mean") else Lambda(lambda x: x)

    _train_transforms = Compose([RandomResizedCrop(size), RandomHorizontalFlip(), ToTensor(), normalize])
    _val_transforms = Compose([Resize(size), CenterCrop(size), ToTensor(), normalize])

    def train_transforms(batch):
        batch["pixel_values"] = [_train_transforms(img.convert("RGB")) for img in batch[data_args.image_column_name]]
        return batch

    def val_transforms(batch):
        batch["pixel_values"] = [_val_transforms(img.convert("RGB")) for img in batch[data_args.image_column_name]]
        return batch

    if training_args.do_train:
        if data_args.max_train_samples:
            dataset["train"] = dataset["train"].shuffle(seed=training_args.seed).select(range(data_args.max_train_samples))
        dataset["train"].set_transform(train_transforms)

    if training_args.do_eval:
        if data_args.max_eval_samples:
            dataset["validation"] = dataset["validation"].shuffle(seed=training_args.seed).select(range(data_args.max_eval_samples))
        dataset["validation"].set_transform(val_transforms)

    def collate_fn(examples):
        return {
            "pixel_values": torch.stack([e["pixel_values"] for e in examples]),
            "labels": torch.tensor([e[data_args.label_column_name] for e in examples]),
        }

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"] if training_args.do_train else None,
        eval_dataset=dataset["validation"] if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        processing_class=image_processor,
        data_collator=collate_fn,
#        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    with mlflow.start_run():
        mlflow.log_params({
            "epochs": str(training_args.num_train_epochs),
            "train_batch_size": training_args.per_device_train_batch_size,
            "eval_batch_size": training_args.per_device_eval_batch_size,
            "seed": training_args.seed,
            "model_name": model_args.model_name_or_path,
        })

        if training_args.do_train:
            train_result = trainer.train()
            trainer.save_model()
            trainer.log_metrics("train", train_result.metrics)
            trainer.save_metrics("train", train_result.metrics)
            trainer.save_state()
            mlflow.log_metrics(train_result.metrics)
            if trainer.state.best_model_checkpoint is not None:
                print("Best model checkpoint: ")
                match = re.search(r"epoch=(\d+)", trainer.state.best_model_checkpoint)
                if match:
                    print("Best model checkpoint: ",int(match.group(1)))
                    mlflow.log_param("best_epoch", int(match.group(1)))


        if training_args.do_eval:
            metrics = trainer.evaluate()
            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)
            mlflow.log_metrics({f"eval_{k}": v for k, v in metrics.items()})

        kwargs = {
            "finetuned_from": model_args.model_name_or_path,
            "tasks": "image-classification",
            "dataset": data_args.dataset_name,
            "tags": ["image-classification", "vision"],
        }
        if training_args.push_to_hub:
            trainer.push_to_hub(**kwargs)
        else:
            trainer.create_model_card(**kwargs)


if __name__ == "__main__":
    main()
