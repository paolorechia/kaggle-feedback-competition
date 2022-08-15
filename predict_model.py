""" Using model to predict using huggingface API"""

from torch import nn, cuda
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
from config import config
from transformers import (
    Trainer
)
import evaluate


num_labels = config.NUM_LABELS

prediction_df = pd.read_csv(config.FP_PREPROCESSED_VAL_CSV)

print("Loading tokenizer and model...", config.FP_TRAINED_MODEL_IN_USE)

from datasets import load_dataset

tokenizer = AutoTokenizer.from_pretrained(config.FP_TRAINED_MODEL_IN_USE)
model = AutoModelForSequenceClassification.from_pretrained(
    config.FP_TRAINED_MODEL_IN_USE, num_labels=config.NUM_LABELS
)

dataset = load_dataset(
    "csv",
    data_files={
        "validation": config.FP_PREPROCESSED_VAL_CSV,
    },
)

dataset["validation"] = dataset["validation"].remove_columns("label")
dataset["validation"] = dataset["validation"].class_encode_column("discourse_effectiveness")
dataset["validation"] = dataset["validation"].rename_column("discourse_effectiveness", "label")


def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=config.TOKENIZER_MAX_SIZE,
    )

print("Tokenizing...")
tokenized_datasets = dataset.map(tokenize_function, batched=True)

trainer = Trainer(
    model=model,
    args=config.training_args,
)
predictions = trainer.predict(tokenized_datasets["validation"])
print(predictions)