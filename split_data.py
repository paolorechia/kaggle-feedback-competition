import numpy as np
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import regex
import math

from sklearn.model_selection import train_test_split
import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from config import config
from utils import discourse_effectiveness_to_int, get_by_category_fp

import re

df = pd.read_csv(config.FP_ORIGINAL_TRAIN_CSV)

stop_words = set(stopwords.words("english"))


def preprocess(text, stop_words_flag=False):
    t = text.lower()
    t = re.sub("\n", "", t)

    t = t.strip()
    t = t.rstrip()

    if stop_words_flag:
        t = re.sub("`", "", t)
        t = re.sub("\.", "", t)
        t = re.sub("'", "", t)
        t = re.sub('"', "", t)
        t = re.sub(",", "", t)
        t = re.sub(":", "", t)
        t = re.sub(";", "", t)
        t = re.sub("\(", "", t)
        t = re.sub("\)", "", t)
        t = t.strip()
        t = t.rstrip()

        word_tokens = word_tokenize(t)
        lowered_tokens = [w.lower() for w in word_tokens]
        filtered_sentence = []
        for token in lowered_tokens:
            if token not in stop_words:
                filtered_sentence.append(token)
        t = str(" ".join(filtered_sentence))
    return t


print(df.columns)
essays = {}
ids_ = list(df.essay_id.unique())
for id_ in ids_:
    path = os.path.join(config.FP_ORIGINAL_TRAIN_ESSAY_DIR, f"{id_}.txt")
    with open(path, "r") as fp:
        text = fp.read()
        essays[id_] = preprocess(text)


train_essay_ids, test_essays_ids = train_test_split(
    ids_, test_size=config.TEST_SIZE, random_state=42
)
train_essay_ids, val_essay_ids = train_test_split(
    train_essay_ids, test_size=config.VAL_SIZE, random_state=42
)


# print(df.text_start)
# print(df.text_start.describe())

df["original_text"] = df.discourse_text
if config.REMOVE_STOP_WORDS:
    df["nltk_preprocessed"] = df.discourse_text.apply(
        lambda x: preprocess(x, stop_words_flag=True)
    )
if config.USE_CONTEXT:

    def f(x):
        p = preprocess(x.discourse_text)
        essay = essays[x.essay_id]
        idx = essay.find(p)
        if idx == -1:
            print("Using fuzzy regex")
            test = regex.search(f"({p})" + "{e<10}", essay)
            if not test:
                return -1
            s = test.start()
            return int(s)
        else:
            return int(idx)

    def get_context_window(x):
        essay = essays[x.essay_id]
        l = len(x.discourse_text)
        type_len = len(x.discourse_type) + 1
        extra_space = config.TOKENIZER_MAX_SIZE - l - type_len
        if extra_space <= 24:
            return l
        front_extra_space = extra_space // 2
        back_extra_space = extra_space // 2
        start_idx = max(0, x.text_start - front_extra_space)
        end_idx = min(len(essay), x.text_start + back_extra_space)

        context_window = ""
        if start_idx < x.text_start:
            context_window = " CSTR " + essay[start_idx : x.text_start] + " CEND "

        context_window += f"{x.discourse_type.upper()} "
        context_window += essay[x.text_start : x.text_start + l]

        if end_idx > x.text_start + l:
            context_window += " CSTR "
            context_window += essay[x.text_start + l : end_idx]
            context_window += " CEND "

        return context_window

    df["text_start"] = df.apply(f, axis=1)
    print(len(df.text_start))
    misses = len(df.text_start[df.text_start == -1])
    assert misses == 0
    df["text_with_context"] = df.apply(get_context_window, axis=1)

df["label"] = df.discourse_effectiveness.apply(discourse_effectiveness_to_int)

df.sort_values(by=["essay_id", "text_start"])


current_essay = None
current_sequence = ""
sequences = []


def discourse_type_to_letter(discourse_type):
    d = {
        "Claim": "A",
        "Concluding Statement": "B",
        "Counterclaim": "C",
        "Evidence": "D",
        "Lead": "E",
        "Position": "F",
        "Rebuttal": "G",
    }
    return d[discourse_type]


def letter_multiplier(letter):
    if letter == "A":
        return 0.0001
    if letter == "B":
        return 0.001
    if letter == "C":
        return 0.01
    if letter == "D":
        return 0.1
    if letter == "E":
        return 1
    if letter == "F":
        return 10
    if letter == "G":
        return 100
    return ValueError(letter)


def parse_sequence_to_number(sequence):
    sum_ = 0.0
    blocks = sequence.split(";")
    # Ignore empty block and ignore our own label (redundant info)
    for block in blocks[0:-2]:
        s = block.split(":")
        letter = s[0]
        label = s[1]
        sum_ += letter_multiplier(letter) * int(label)
    return sum_


def extract_last_label(sequence):
    blocks = sequence.split(";")
    # Ignore empty block and ignore our own label (redundant info)
    for block in blocks[0:-2]:
        s = block.split(":")
        label = s[1]
        return label
    return -1


for idx, row in df.iterrows():
    if row.essay_id != current_essay:
        current_essay = row.essay_id
        current_sequence = ""
    sequence_letter = discourse_type_to_letter(row.discourse_type)
    sequence_number = row.label
    current_sequence += f"{sequence_letter}:{sequence_number};"
    sequences.append(current_sequence)

df["sequence_code"] = sequences
df["sequence_sum"] = df.sequence_code.apply(parse_sequence_to_number)
df["previous_label"] = df.sequence_code.apply(extract_last_label)
print(df.head(n=20))
print(df.tail(n=20))


base_train_df = df[df.essay_id.isin(train_essay_ids)]
base_test_df = df[df.essay_id.isin(test_essays_ids)]
base_val_df = df[df.essay_id.isin(val_essay_ids)]


base_train_df = base_train_df.drop("essay_id", axis=1)
base_test_df = base_test_df.drop("essay_id", axis=1)
base_val_df = base_val_df.drop("essay_id", axis=1)


num_labels = len(df.discourse_effectiveness.unique())
# df = df.rename(columns={"discourse_text": "text"})
print("Splitting by category")
discourse_types = df.discourse_type.unique()
print(discourse_types)
for type_ in discourse_types:
    train_category_df = base_train_df[base_train_df.discourse_type == type_]
    test_category_df = base_test_df[base_test_df.discourse_type == type_]
    val_category_df = base_val_df[base_val_df.discourse_type == type_]

    print(type_, len(train_category_df), len(test_category_df), len(val_category_df))

    train_fp = get_by_category_fp(
        config.FP_PREPROCESSED_BY_CATEGORY_CSV_DIR, "train", type_
    )
    test_fp = get_by_category_fp(
        config.FP_PREPROCESSED_BY_CATEGORY_CSV_DIR, "test", type_
    )
    val_fp = get_by_category_fp(
        config.FP_PREPROCESSED_BY_CATEGORY_CSV_DIR, "val", type_
    )
    train_category_df.to_csv(train_fp, index=False)
    test_category_df.to_csv(test_fp, index=False)
    val_category_df.to_csv(val_fp, index=False)

print("Using full dataset")

discourse_types = df.discourse_type.unique()
discourse_effectiveness_types = df.discourse_effectiveness.unique()
print(discourse_types)
for main_df, df_name in [
    (df, "Full dataset"),
    (base_train_df, "Train dataset"),
    (base_test_df, "Test dataset"),
    (base_val_df, "Validation dataset"),
]:
    print("===========", df_name, "============")
    for type_ in discourse_types:
        category_df = main_df[main_df.discourse_type == type_]
        print(type_, len(category_df))
        for effectiveness in discourse_effectiveness_types:
            effectiveness_df = category_df[
                category_df.discourse_effectiveness == effectiveness
            ]
            print("-----> ", effectiveness, len(effectiveness_df))

# Fullset
base_train_df.to_csv(config.FP_PREPROCESSED_TRAIN_CSV, index=False)
base_test_df.to_csv(config.FP_PREPROCESSED_TEST_CSV, index=False)
base_val_df.to_csv(config.FP_PREPROCESSED_VAL_CSV, index=False)
