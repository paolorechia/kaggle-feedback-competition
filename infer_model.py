# Using model to predict
from torch import nn, cuda
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
from config import config

prediction_df = pd.read_csv(config.FP_ORIGINAL_TEST_CSV)
prediction_df = prediction_df.rename(columns={"discourse_text": "text"})

inputs_ = list(prediction_df.text)


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

print("Going to start predicting... ")
for text in inputs_:
    pt_batch = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=config.TOKENIZER_MAX_SIZE,
        return_tensors="pt",
    )
    pt_outputs = model(**pt_batch)
    pt_predictions = (
        nn.functional.softmax(pt_outputs.logits, dim=-1).detach().to("cpu").numpy()
    )
    #     print("Prediction: ", pt_predictions)
    results["Ineffective"].append(pt_predictions[0][0])
    results["Adequate"].append(pt_predictions[0][1])
    results["Effective"].append(pt_predictions[0][2])

    del pt_batch
    cuda.empty_cache()

print("Finished predicting!")

# print("Id to label mapping ", model.config.id2label)
# Need to figure out how to get the label indices!!
# print(results)
prediction_df["Ineffective"] = results["Ineffective"]
prediction_df["Adequate"] = results["Adequate"]
prediction_df["Effective"] = results["Effective"]
# print(prediction_df.columns)
prediction_df = prediction_df.drop("essay_id", axis=1)
prediction_df = prediction_df.drop("discourse_type", axis=1)
prediction_df = prediction_df.drop("text", axis=1)
print(prediction_df.head())

prediction_df.to_csv(
    config.FP_OUTPUT_SAMPLE_SUBMISSION_CSV, index=False, float_format="%.1f"
)
