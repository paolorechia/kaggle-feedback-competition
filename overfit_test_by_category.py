import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
)
from datasets import load_dataset, load_metric
import torch
import os
from config import config
import json
import utils

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


metric = load_metric(config.METRIC)
results = {}
print("Overfit by category for model... ", config.MODEL_NAME_IN_USE)
for category in utils.categories:
    print(category)
    print("Calling load dataset...")
    dataset = load_dataset(
        "csv",
        data_files={
            "train": utils.get_by_category_fp(
                config.FP_PREPROCESSED_BY_CATEGORY_CSV_DIR, "train", category
            ),
        },
    )

    dataset["train"] = dataset["train"].remove_columns("label")
    dataset["train"] = dataset["train"].class_encode_column("discourse_effectiveness")
    dataset["train"] = dataset["train"].rename_column(
        "discourse_effectiveness", "label"
    )

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
    # print(tokenized_datasets)

    trainer = Trainer(
        model=model,
        args=config.training_args,
        train_dataset=tokenized_datasets["train"],
        compute_metrics=compute_metrics,
    )

    # Let's overfit
    # Uncomment the block below to test if the model overfits with a single batch

    print("Let's overfit to test the model...")
    device = "cuda"
    for batch in trainer.get_train_dataloader():
        break

    print("Sending batch to GPU...")
    batch = {k: v.to(device) for k, v in batch.items()}
    trainer.create_optimizer()

    print("Sending model to GPU...")
    trainer.model.cuda()
    trainer.model.train()
    print("Overfitting...")
    for _ in range(20):
        outputs = trainer.model(**batch)
        loss = outputs.loss
        loss.backward()
        trainer.optimizer.step()
        trainer.optimizer.zero_grad()

    print("Evaluating overfitting test...")
    trainer.model.eval()
    with torch.no_grad():
        outputs = trainer.model(**batch)
    preds = outputs.logits
    labels = batch["labels"]

    result = compute_metrics((preds.cpu().numpy(), labels.cpu().numpy()))

    print(result)
    results[category] = result

for key, item in results.items():
    print(key, item)
