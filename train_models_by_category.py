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
import json
import uuid


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


metric = load_metric(config.METRIC)

categories = utils.categories
experiment_id = uuid.uuid4()

# Modify this to train other categories
training_categories = ["Rebuttal"]
categories = training_categories

datasets = {}
print("Calling load datasets...")
for category in categories:
    datasets[category] = load_dataset(
        "csv",
        data_files={
            "train": utils.get_by_category_fp(
                config.FP_PREPROCESSED_BY_CATEGORY_CSV_DIR, "train", category
            ),
            "test": utils.get_by_category_fp(
                config.FP_PREPROCESSED_BY_CATEGORY_CSV_DIR, "test", category
            ),
        },
    )

    dataset = datasets[category]
    dataset["train"] = dataset["train"].remove_columns("label")
    dataset["train"] = dataset["train"].class_encode_column("discourse_effectiveness")
    dataset["train"] = dataset["train"].rename_column(
        "discourse_effectiveness", "label"
    )

    dataset["test"] = dataset["test"].class_encode_column("discourse_effectiveness")
    dataset["test"] = dataset["test"].remove_columns("label")
    dataset["test"] = dataset["test"].rename_column("discourse_effectiveness", "label")

    print("Labels")
    for partition in ["train", "test"]:
        print(dataset[partition].features["label"].names)
        raw_train_dataset = dataset[partition]
        print(raw_train_dataset.features)
        # print(raw_train_dataset[0])

        with open(f"labels-{partition}-{category}.json", "w") as fp:
            json.dump(dataset[partition].features["label"].names, fp)


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

os.environ["WANDB_DISABLED"] = "true"
print(tokenized_datasets)

for category in categories:
    model = AutoModelForSequenceClassification.from_pretrained(
        config.MODEL_NAME_IN_USE, num_labels=config.NUM_LABELS
    )
    print("Overriding model config...")
    model.config = config.override_deberta_v2_config(model)

    print("Model config", model.config)
    trainer = Trainer(
        model=model,
        args=config.training_parameters_per_category[category],
        train_dataset=tokenized_datasets[category]["train"],
        eval_dataset=tokenized_datasets[category]["test"],
        compute_metrics=compute_metrics,
    )

    print(
        f"About to start training model {config.MODEL_NAME_IN_USE} on category: {category}"
    )
    model_output_dir = os.path.join(
        config.FP_TRAINED_MODEL_IN_USE, "by_category", category
    )
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

    print("Evaluating trained model...")
    result = trainer.evaluate()
    print(result)
    to_save = {"result": result, "args": config.experimental_args[category]}

    print("Saving results...")
    filename = f"{category}_{experiment_id}.json"
    path_ = os.path.join(config.FP_EXPERIMENT_PARAMETERS_DIR, filename)
    with open(path_, "w") as fp:
        json.dump(to_save, fp)
    print("Saved!")
