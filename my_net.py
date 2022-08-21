import torch
import torch.nn as nn
from config import config
import utils
import pandas as pd
import numpy as np
import json


def load_dataset(partition="train", category="Claim"):
    label_fp = f"fast_y_{partition}_{category}.json"
    data_fp = f"fast_x_{partition}_{category}.json"
    try:
        with open(label_fp, "r") as fp:
            y = np.array(json.load(fp))
        with open(data_fp, "r") as fp:
            X = np.array(json.load(fp), dtype=np.float32)
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


def create_tensors(X, y):
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
        self.linear1 = nn.Linear(38400, 200, dtype=torch.float)
        self.linear2 = nn.Linear(200, 3)

    def forward(self, x):
        h = torch.relu(self.linear1(x))
        o = torch.relu(self.linear2(h))
        return o


print("Using torch device", device)
print("Moving model to device...")
model = Model()
model.to(device)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

print("Loading data...")
category = "Claim"
feature_column = "flatten_text_matrix"


X_train, y_train = load_dataset(partition="train")

X_train_tensor, y_train_tensor = create_tensors(X_train, y_train)

print("Training...")
## y in math
for epoch in range(10):
    for idx in range(len(X_train_tensor)):
        X = X_train_tensor[idx].to(device)
        y = y_train_tensor[idx].to(device)
        optimizer.zero_grad()
        result = model(X)
        loss = criterion(result, y)
        loss.backward()
        optimizer.step()
    print("epoch {}, loss {}".format(epoch, loss.item()))
