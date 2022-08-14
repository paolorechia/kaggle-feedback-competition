# Using model to predict
from torch import nn, cuda
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import log_loss

from datetime import datetime
import pandas as pd
from config import config
import utils

prediction_df = pd.read_csv(config.FP_PREPROCESSED_VAL_CSV)

infer_subset_size = 100
infer_subset_size = len(prediction_df)

prediction_df = prediction_df[0:infer_subset_size] 

inputs_ = list(prediction_df.text)
labels = list(prediction_df.discourse_effectiveness)

results = {
    "Ineffective": [],
    "Adequate": [],
    "Effective": [],
}
print("Loading tokenizer and model...", config.FP_TRAINED_MODEL_IN_USE)

num_labels = config.NUM_LABELS

tokenizer = AutoTokenizer.from_pretrained(config.FP_TRAINED_MODEL_IN_USE)
model = AutoModelForSequenceClassification.from_pretrained(
    config.FP_TRAINED_MODEL_IN_USE, num_labels=config.NUM_LABELS
)
print("Sending model to GPU")
model.cuda()

print("Going to start predicting... ")
hits = 0
miss = 0
total = 0
accuracy = 0
loss_sum = 0.0
running_loss = 0.0

t0 = datetime.now()
for j, text in enumerate(inputs_):
    pt_batch = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=config.TOKENIZER_MAX_SIZE,
        return_tensors="pt",
    )
    print("Sending batch to GPU")
    pt_batch.to("cuda")
    pt_outputs = model(**pt_batch)
    pt_predictions = (
        nn.functional.softmax(pt_outputs.logits, dim=-1).detach().to("cpu").numpy()
    )

    current_label = labels[j]
    target = [0.0, 0.0, 0.0]
    if current_label == "Ineffective":
        target[0] = 1.0
    elif current_label == "Adequate":
        target[1] = 1.0
    elif current_label == "Effective":
        target[2] = 1.0
    else:
        ValueError("Invalid label")

    # loss = criterion(pt_outputs, current_label)
    print(pt_predictions, target)
    loss = log_loss(target, pt_predictions[0])
    print("Current loss", loss)
    loss_sum += loss
    running_loss = loss_sum / (j + 1)
    print("Running Loss: ", running_loss)

    #     print("Prediction: ", pt_predictions)
    results["Ineffective"].append(pt_predictions[0][0])
    results["Adequate"].append(pt_predictions[0][1])
    results["Effective"].append(pt_predictions[0][2])
    
    print("Predictions: ", pt_predictions)
    max_ = -100
    max_i = 0
    for i in range(len(pt_predictions[0])):
        if pt_predictions[0][i] > max_:
            max_i = i
            max_ = pt_predictions[0][i]

    actual_label = utils.label_number_to_label_name(max_i)
    target_label = labels[j]

    print(f"Guessed: {actual_label} - Actual: {target_label}")
    if actual_label == target_label:
        hits += 1
    else:
        miss += 1
    del pt_batch
    cuda.empty_cache()
    total = hits + miss
    accuracy = hits / total
    print(f"At row {j}: accuracy - {accuracy}") 

t1 = datetime.now()

print("Finished predicting!")
print(f"Accuracy: {accuracy}")
print(f"Running loss: {running_loss}")

delta = t1 - t0
diff_in_seconds = delta.seconds
time_per_inferece = diff_in_seconds / infer_subset_size
print(f"Elapsed inference time: {diff_in_seconds} seconds")
print(f"Required time per inference: {time_per_inferece} seconds")

prediction_df["Ineffective"] = results["Ineffective"]
prediction_df["Adequate"] = results["Adequate"]
prediction_df["Effective"] = results["Effective"]

prediction_df = prediction_df.drop("discourse_type", axis=1)
prediction_df = prediction_df.drop("text", axis=1)
print(prediction_df.head())

prediction_df.to_csv(
    config.FP_OUTPUT_SAMPLE_SUBMISSION_CSV, index=False, float_format="%.1f"
)
