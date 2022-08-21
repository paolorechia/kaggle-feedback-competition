import os
from config import config
import numpy as np


def discourse_effectiveness_to_int(discourse_effectiveness):
    l = discourse_effectiveness.lower()
    if l == "ineffective":
        return 0
    if l == "adequate":
        return 1
    if l == "effective":
        return 2
    raise ValueError(f"Unrecognized discourse effectiveness: {discourse_effectiveness}")


# Not used, but still useful to know
def label_number_to_label_name(x):
    if x == 0:
        return "Ineffective"
    if x == 1:
        return "Adequate"
    if x == 2:
        return "Effective"
    raise ValueError("Unknown label")


# Not used, but still useful to know
def label_number_to_label_name_deberta_v3_base(x):
    if x == 2:
        return "Ineffective"
    if x == 0:
        return "Adequate"
    if x == 1:
        return "Effective"
    raise ValueError("Unknown label")


def get_by_category_fp(base_path, subset_partition: str, category):
    filename = f"{subset_partition}_{category}.csv"
    return os.path.join(base_path, filename)


categories = [
    "Claim",
    "Concluding Statement",
    "Counterclaim",
    "Evidence",
    "Lead",
    "Position",
    "Rebuttal",
]

import json

flattened_cache = {}


def parse_and_pad_flattened_arrays(X, cache_label=None):
    if cache_label in flattened_cache:
        return np.array(flattened_cache[cache_label])
    if cache_label is None:
        print("PLEASE SET A CACHE LABEL, THIS IS PAINFULLY SLOW")
    try:
        with open(f"{cache_label}.json", "r") as fp:
            cached = json.load(fp)
            print("Loaded from cache...")
            flattened_cache[cache_label] = cached
            return np.array(cached)
    except Exception:
        pass

    print("Cache miss!")
    desired_length = config.FLATTEN_MAX_LENGTH * config.FAST_TEXT_EMBEDDING_SIZE
    X_new = []
    lists = X.apply(lambda x: eval(x))
    for idx, item in lists.iteritems():
        l = len(item)
        if l > desired_length:
            item = item[0:l]
        while l < desired_length:
            item.append(0.0)
            l = len(item)
        X_new.append(item)
    if cache_label:
        print("Saving to cache...")
        with open(f"{cache_label}.json", "w") as fp:
            json.dump(X_new, fp)
            flattened_cache[cache_label] = np.array(X_new)
            return flattened_cache[cache_label]
    return np.array(X_new)
