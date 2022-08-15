import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
)
from datasets import load_dataset, load_metric
from torch import nn
import torch
import os
from config import config
import sys
import json


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
# print(tokenized_datasets["train"])
# print(tokenized_datasets["train"][0])

# print(tokenizer.decode(tokenized_datasets["train"][0]["input_ids"]))
# sample_test = """Lead Hi, i'm Isaac, i'm going to be writing about how this face on Mars is a natural landform or if there is life on Mars that made it. The story is about how NASA took a picture of Mars and a face was seen on the planet. NASA doesn't know if the landform was created by life on Mars, or if it is just a natural landform.  Hi, i'm Isaac, i'm going to be writing about how this face on Mars is a natural landform or if there is life on Mars that made it. The story is about how NASA took a picture of Mars and a face was seen on the planet. NASA doesn't know if the landform was created by life on Mars, or if it is just a natural landform. On my perspective, I think that the face is a natural landform because I dont think that there is any life on Mars. In these next few paragraphs, I'll be talking about how I think that is is a natural landform

# I think that the face is a natural landform because there is no life on Mars that we have descovered yet. If life was on Mars, we would know by now. The reason why I think it is a natural landform because, nobody live on Mars in order to create the figure. It says in paragraph 9, ""It's not easy to target Cydonia,"" in which he is saying that its not easy to know if it is a natural landform at this point. In all that they're saying, its probably a natural landform.

# People thought that the face was formed by alieans because they thought that there was life on Mars. though some say that life on Mars does exist, I think that there is no life on Mars.

# It says in paragraph 7, on April 5, 1998, Mars Global Surveyor flew over Cydonia for the first time. Michael Malin took a picture of Mars with his Orbiter Camera, that the face was a natural landform. Everyone who thought it was made by alieans even though it wasn't, was not satisfied. I think they were not satisfied because they have thought since 1976 that it was really formed by alieans.

# Though people were not satified about how the landform was a natural landform, in all, we new that alieans did not form the face. I would like to know how the landform was formed. we know now that life on Mars doesn't exist.             ",1"""

# print(tokenizer.encode(sample_test))

# sys.exit(0)
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

# Let's overfit
# Uncomment the block below to test if the model overfits with a single batch

# print("Let's overfit to test the model...")
# device = "cuda"
# for batch in trainer.get_train_dataloader():
#     break

# print("Sending batch to GPU...")
# batch = {k: v.to(device) for k, v in batch.items()}
# trainer.create_optimizer()

# print("Sending model to GPU...")
# trainer.model.cuda()
# trainer.model.train()
# print("Overfitting...")
# for _ in range(20):
#     outputs = trainer.model(**batch)
#     loss = outputs.loss
#     loss.backward()
#     trainer.optimizer.step()
#     trainer.optimizer.zero_grad()

# print("Evaluating overfitting test...")
# trainer.model.eval()
# with torch.no_grad():
#     outputs = trainer.model(**batch)
# preds = outputs.logits
# labels = batch["labels"]

# result = compute_metrics((preds.cpu().numpy(), labels.cpu().numpy()))

# print(result)
# sys.exit(0)

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
# Epoch 1, test 20%
# {'eval_loss': 0.7537392377853394, 'eval_accuracy': 0.6627226982184142, 'eval_runtime': 61.4557, 'eval_samples_per_second': 119.647, 'eval_steps_per_second': 14.97, 'epoch': 0.54}
# Epoch 1, test 50%
# {'eval_loss': 0.7370373606681824, 'eval_accuracy': 0.6702932056791601, 'eval_runtime': 148.8945, 'eval_samples_per_second': 123.463, 'eval_steps_per_second': 3.862, 'epoch': 0.96}
