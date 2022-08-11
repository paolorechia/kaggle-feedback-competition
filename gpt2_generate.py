from inspect import GEN_CLOSED
from termios import N_SLIP
from transformers import pipeline
from utils import categories, discourse_effectiveness_to_int
import pandas as pd
from config import config
import os
import re

writer = pipeline(
    "text-generation", model="./trained_models/gpt2-text-generation", tokenizer="gpt2"
)

N_SAMPLES = 10000
generated_texts = []
generated_effectiveness = []

category = "Position"
output_fp = os.path.join(config.FP_GENERATED_DIR, f"{category}.csv")
try:
    os.makedirs(config.FP_GENERATED_DIR)
except FileExistsError:
    pass

print(f"Will generate {N_SAMPLES} of type '{category}' and save to {output_fp}")

effectiveness_to_generate = ["Ineffective", "Adequate", "Effective"]

for effectiveness in effectiveness_to_generate:
    for i in range(N_SAMPLES):
        gen_hint = f"{category} {effectiveness}:"
        result = writer(gen_hint)[0]["generated_text"]
        trimmed = result.split(gen_hint)[1]
        first_sentence = re.sub("\n", "", trimmed.split(".")[0] + ".")
        if len(first_sentence) > 10:
            generated_texts.append(first_sentence)
            generated_effectiveness.append(effectiveness)

print("Generated...")
# for i in range(len(generated_texts)):
#     print(generated_effectiveness[i], generated_texts[i])

print("Saving...")
generated_df = pd.DataFrame(
    {
        "essay_id": "generated",
        "discourse_effectiveness": generated_effectiveness,
        "discourse_text": generated_texts,
    }
)
generated_df["label"] = generated_df.discourse_effectiveness.apply(
    discourse_effectiveness_to_int
)

generated_df.to_csv(output_fp)
print("Success!")