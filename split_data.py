from random import random
from unicodedata import category
import numpy as np
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split
import os

from config import config
from utils import discourse_effectiveness_to_int, get_by_category_fp

df = pd.read_csv(config.FP_ORIGINAL_TRAIN_CSV)

print(df.columns)
essays = {}
ids_ = list(df.essay_id.unique())
for id_ in ids_:
    path = os.path.join(config.FP_ORIGINAL_TRAIN_ESSAY_DIR, f"{id_}.txt")
    with open(path, "r") as fp:
        text = fp.read()
        essays[id_] = text


train_essay_ids, test_essays_ids = train_test_split(
    ids_, test_size=config.TEST_SIZE, random_state=42
)
train_essay_ids, val_essay_ids = train_test_split(
    train_essay_ids, test_size=config.VAL_SIZE, random_state=42
)

df["text"] = df.apply(
    lambda x: x.discourse_type + " " + x.discourse_text + " " + essays[x.essay_id],
    axis=1,
)
df["label"] = df.discourse_effectiveness.apply(discourse_effectiveness_to_int)

base_train_df = df[df.essay_id.isin(train_essay_ids)]
base_test_df = df[df.essay_id.isin(test_essays_ids)]
base_val_df = df[df.essay_id.isin(val_essay_ids)]



base_train_df = base_train_df.drop("essay_id", axis=1)
base_test_df = base_test_df.drop("essay_id", axis=1)
base_val_df = base_val_df.drop("essay_id", axis=1)


num_labels = len(df.discourse_effectiveness.unique())
# df = df.rename(columns={"discourse_text": "text"})

if config.SPLIT_BY_CATEGORY:
    # TODO: split by essay here too
    print("Splitting by category")
    discourse_types = df.discourse_type.unique()
    print(discourse_types)
    for type_ in discourse_types:
        train_category_df = base_train_df[base_train_df.discourse_type == type_]
        test_category_df = base_test_df[base_test_df.discourse_type == type_]
        val_category_df = base_val_df[base_val_df.discourse_type == type_]

        print(
            type_, len(train_category_df), len(test_category_df), len(val_category_df)
        )

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

else:
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
