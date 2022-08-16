import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
)
from datasets import load_dataset, load_metric
import os
from config import config
import json

import torch
import random

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


metric = load_metric(config.METRIC)

print("Calling load dataset...")
dataset = load_dataset(
    "csv",
    data_files={
        "train": config.FP_PREPROCESSED_TRAIN_CSV,
        "test": config.FP_PREPROCESSED_TEST_CSV,
    },
)

dataset["train"] = dataset["train"].remove_columns("label")
dataset["train"] = dataset["train"].class_encode_column("discourse_effectiveness")
dataset["train"] = dataset["train"].rename_column("discourse_effectiveness", "label")

dataset["test"] = dataset["test"].class_encode_column("discourse_effectiveness")
dataset["test"] = dataset["test"].remove_columns("label")
dataset["test"] = dataset["test"].rename_column("discourse_effectiveness", "label")

print(dataset["train"].features)
print(dataset["test"].features)

print("Labels")
print(dataset["train"].features["label"].names)
raw_train_dataset = dataset["train"]
print(raw_train_dataset.features)
print(raw_train_dataset[0])

with open("labels.json", "w") as fp:
    json.dump(dataset["train"].features["label"].names, fp)

print("Import tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME_IN_USE)


def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=config.TOKENIZER_MAX_SIZE,
    )


print("Tokenizing...")
tokenized_datasets = dataset.map(tokenize_function, batched=True)

model = AutoModelForSequenceClassification.from_pretrained(
    config.MODEL_NAME_IN_USE, num_labels=config.NUM_LABELS
)

os.environ["WANDB_DISABLED"] = "true"
print(tokenized_datasets)

trainer = Trainer(
    model=model,
    args=config.training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    compute_metrics=compute_metrics,
)


print("About to start training model... ", config.MODEL_NAME_IN_USE)

print("Arguments", config.training_args)
model_output_dir = config.FP_TRAINED_MODEL_IN_USE
print("Will save results to: ", model_output_dir)

print("Started!")
trainer.train()
print("Finished! Saving...")

trainer.save_model(model_output_dir)
tokenizer.save_pretrained(model_output_dir)

print("Evaluating trained model...")
result = trainer.evaluate()
print(result)
