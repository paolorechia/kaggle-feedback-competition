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

print(dataset)

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
model_output_dir = config.FP_TRAINED_MODEL_IN_USE
print("Will save results to: ", model_output_dir)

print("Started!")
trainer.train()
print("Finished! Saving...")

trainer.save_model(model_output_dir)
tokenizer.save_pretrained(model_output_dir)

# Epoch 1, test 20%
# {'eval_loss': 0.7537392377853394, 'eval_accuracy': 0.6627226982184142, 'eval_runtime': 61.4557, 'eval_samples_per_second': 119.647, 'eval_steps_per_second': 14.97, 'epoch': 0.54}
# Epoch 1, test 50%
# {'eval_loss': 0.7370373606681824, 'eval_accuracy': 0.6702932056791601, 'eval_runtime': 148.8945, 'eval_samples_per_second': 123.463, 'eval_steps_per_second': 3.862, 'epoch': 0.96}
