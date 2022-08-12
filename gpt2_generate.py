from transformers import pipeline
from utils import categories, discourse_effectiveness_to_int
import pandas as pd
from config import config
import os
import re
from datetime import datetime
import torch
import utils

writer = pipeline(
    "text-generation", model="./trained_models/gpt2-text-generation", tokenizer="gpt2"
)

N_SAMPLES = 1000 # Per class
BATCH_SIZE = 100



for category in utils.categories:
    generated_texts = []
    generated_effectiveness = []
    output_fp = os.path.join(config.FP_GENERATED_DIR, f"{category}{N_SAMPLES * 3}.csv")
    try:
        os.makedirs(config.FP_GENERATED_DIR)
    except FileExistsError:
        pass

    print(f"{datetime.now()}: Will generate {N_SAMPLES * 3} samples of type '{category}' and save to {output_fp}")

    effectiveness_to_generate = ["Ineffective", "Adequate", "Effective"]
    device = torch.device("cuda")

    batches = int(N_SAMPLES / BATCH_SIZE)
    sequence_number = int(min(N_SAMPLES, BATCH_SIZE))


    for idx in range(batches):
        print(f"{datetime.now()}: Working on batch: {idx + 1} out of {batches}...")
        for effectiveness in effectiveness_to_generate:
            print(f"{datetime.now()}: Generating effectiveness: {effectiveness}")
            gen_hint = f"{category} {effectiveness}:"
            result_list = writer(
                gen_hint,
                max_length=512,
                num_return_sequences=sequence_number,
                device=device,
            )
            trimmed_sentences = [
                re.sub(
                    "\n",
                    "",
                    result["generated_text"].split(gen_hint)[1].split(".")[0] + ".",
                )
                for result in result_list
            ]
            for sentence in trimmed_sentences:
                # Clean sentence
                for category in categories:
                    if category in sentence:
                        sentence = sentence.split(category)[0]
                if "Essay" in sentence:
                    sentence = sentence.split("Essay")[0]
                if len(sentence) > 10:
                    generated_texts.append(sentence)
                    generated_effectiveness.append(effectiveness)

        print(f"{datetime.now()}: Generated batch...")
        print(f"{datetime.now()}: Saving...")

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
        generated_df.to_csv(output_fp, index=False)
        print(f"{datetime.now()}: Generated samples so far: {len(generated_df)}")

print(f"{datetime.now()}: Success!")
