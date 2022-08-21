import torch
import torch.nn as nn
from config import config
import utils
import pandas as pd
import numpy as np
import json
import random


torch.manual_seed(0)
random.seed(0)
np.random.seed(0)


def load_dataset(partition="train", category="Claim"):
    label_fp = f"fast_y_{partition}_{category}.json"
    data_fp = f"fast_x_{partition}_{category}.json"
    try:
        with open(label_fp, "r") as fp:
            y = np.array(json.load(fp))
        with open(data_fp, "r") as fp:
            X = np.array(json.load(fp), dtype=np.float16)
        print("Loaded data from cache L2")
    except:
        csv_fp = utils.get_by_category_fp(
            config.FP_PREPROCESSED_BY_CATEGORY_CSV_DIR, partition, category
        )
        df = pd.read_csv(csv_fp, usecols=["label", "flatten_text_matrix"])
        y = df.label
        X = df[feature_column]
        if feature_column == "flatten_text_matrix":
            X = utils.parse_and_pad_flattened_arrays(X, f"{category}-{partition}")
        with open(label_fp, "w") as fp:
            json.dump(list(y), fp)
        with open(data_fp, "w") as fp:
            main_list = []
            for l in list(X):
                main_list.append(list(l))
            json.dump(main_list, fp)
    return X, y


def create_tensors(X, y, use_bfloat16=False):
    print("Converting to tensor...")
    print(X.shape)
    X_tensor = torch.from_numpy(X)
    print(X_tensor.shape)
    print(X_tensor[0].shape)
    print(X_tensor[0])

    labels_as_prob = []
    print("Creating labels tensor...")
    for label in y:
        labels = [0.0, 0.0, 0.0]
        labels[label] += 1.0
        labels_as_prob.append(labels)

    y_tensor = torch.tensor(labels_as_prob)
    print(y_tensor.shape)
    print(y_tensor[0].shape)
    print(y_tensor[0])
    print("Sending tensors to GPU")
    X_tensor.cuda()
    y_tensor.cuda()
    if use_bfloat16:
        return X_tensor.bfloat16(), y_tensor.bfloat16()
    return X_tensor, y_tensor


if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"
device = torch.device(device)

print("Defining network...")


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(38400, 1000, dtype=torch.bfloat16)
        self.linear2 = nn.Linear(1000, 200, dtype=torch.bfloat16)
        self.linear3 = nn.Linear(200, 3, dtype=torch.bfloat16)

    def forward(self, x):

        a = torch.relu(self.linear1(x))
        b = torch.relu(self.linear2(a))
        c = torch.relu(self.linear3(b))
        return c


print("Using torch device", device)
print("Moving model to device...")
model = Model()
model.to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
criterion = nn.CrossEntropyLoss()
print("Loading data...")
category = "Claim"
feature_column = "flatten_text_matrix"


X_train, y_train = load_dataset(partition="train")
X_train_tensor, y_train_tensor = create_tensors(X_train, y_train, use_bfloat16=True)

print("Training...")
## y in math
total_loss = 0.0
j = 0
for epoch in range(30):
    for idx in range(len(X_train_tensor)):
        X = X_train_tensor[idx].to(device)
        y = y_train_tensor[idx].to(device)
        optimizer.zero_grad()
        result = model(X)
        loss = criterion(result, y)
        loss.backward()
        total_loss += loss.item()
        optimizer.step()
        j += 1
    print("epoch {}, loss {}".format(epoch, total_loss / j))

print("Cleaning training data...")
del X_train
del y_train
del X_train_tensor
del y_train_tensor


def test_model_on(partition):
    print("Testing on partition... ", partition)
    X, y = load_dataset(partition)
    X_tensor, y_tensor = create_tensors(X, y, use_bfloat16=True)

    hits = 0.0
    misses = 0.0
    total_loss = 0.0
    j = 0
    for idx in range(len(X_tensor)):
        X = X_tensor[idx].to(device)
        y = y_tensor[idx].to(device)
        with torch.no_grad():
            result = model(X)
            loss = criterion(result, y)
            actual_class = torch.argmax(y)
            predicted_class = torch.argmax(result)
            if actual_class == predicted_class:
                hits += 1
            else:
                misses += 1
            total_loss += loss.item()
    print("Loss {}".format(total_loss / len(X_tensor)))
    accuracy = hits / (hits + misses)
    print(f"Model accuracy in {partition} dataset: {accuracy}")

    print("Cleaning test data...")
    del X
    del y
    del X_tensor
    del y_tensor


test_model_on("train")
test_model_on("test")
test_model_on("val")
