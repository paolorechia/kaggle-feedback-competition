import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
)
from datasets import load_dataset, load_metric
from torch import nn
import os
from config import config
import utils

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


metric = load_metric(config.METRIC)

categories = utils.categories
datasets = {}
print("Calling load datasets...")
for category in categories:
    datasets[category] = load_dataset(
        "csv",
        data_files={
            "train": utils.get_by_category_fp(config.FP_PREPROCESSED_BY_CATEGORY_CSV_DIR, "train", category),
            "test": utils.get_by_category_fp(config.FP_PREPROCESSED_BY_CATEGORY_CSV_DIR, "test", category),
        },
)

print(datasets)

print("Import tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME_IN_USE)


def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=config.TOKENIZER_MAX_SIZE,
    )


print("Tokenizing datasets...")
tokenized_datasets = {}
for category, dataset in datasets.items():
    tokenized_datasets[category] = dataset.map(tokenize_function, batched=True)

model = AutoModelForSequenceClassification.from_pretrained(
    config.MODEL_NAME_IN_USE, num_labels=config.NUM_LABELS
)

os.environ["WANDB_DISABLED"] = "true"
print(tokenized_datasets)

for category in categories:
    trainer = Trainer(
        model=model,
        args=config.training_parameters_per_category[category],
        train_dataset=tokenized_datasets[category]["train"],
        eval_dataset=tokenized_datasets[category]["test"],
        compute_metrics=compute_metrics,
    )

    print(f"About to start training model {config.MODEL_NAME_IN_USE} on category: {category}")
    model_output_dir = os.path.join(config.FP_TRAINED_MODEL_IN_USE, "by_category", category)
    print("Will save results to: ", model_output_dir)

    try:
        os.makedirs(model_output_dir)
    except FileExistsError:
        pass


    print("Started!")
    trainer.train()
    print("Finished! Saving...")

    trainer.save_model(model_output_dir)
    tokenizer.save_pretrained(model_output_dir)