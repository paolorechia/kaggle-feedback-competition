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
num_labels = len(df.discourse_effectiveness.unique())

df["label"] = df.discourse_effectiveness.apply(discourse_effectiveness_to_int)
df = df.drop("essay_id", axis=1)
df = df.rename(columns={"discourse_text": "text"})

if config.SPLIT_BY_CATEGORY:
    print("Splitting by category")
    discourse_types = df.discourse_type.unique()
    print(discourse_types)
    for type_ in discourse_types:
        category_df = df[df.discourse_type == type_]
        print(type_, len(category_df))
        train, test = train_test_split(category_df, test_size=config.TEST_SIZE, random_state=42)
        train_fp = get_by_category_fp(config.FP_PREPROCESSED_BY_CATEGORY_CSV_DIR, "train", type_)
        test_fp = get_by_category_fp(config.FP_PREPROCESSED_BY_CATEGORY_CSV_DIR, "test", type_)
        train.to_csv(train_fp, index=False)
        test.to_csv(test_fp, index=False)        

else:
    print("Using full dataset")
    train, test = train_test_split(df, test_size=config.TEST_SIZE, random_state=42)
    if config.USE_SMALL_DATASET:
        # Test with 10
        train = train.head(n=10)
        test = test.head(n=10)

    discourse_types = df.discourse_type.unique()
    discourse_effectiveness_types = df.discourse_effectiveness.unique()
    print(discourse_types)
    for main_df, df_name in [
        (df, "Full dataset"),
        (train, "Train dataset"),
        (test, "Test dataset"),
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
    train.to_csv(config.FP_PREPROCESSED_TRAIN_CSV, index=False)
    test.to_csv(config.FP_PREPROCESSED_TEST_CSV, index=False)
