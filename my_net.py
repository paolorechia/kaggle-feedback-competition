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


def create_tensors(X, y, use_bfloat16=False, as_matrix=True):
    print("Converting to tensor...")
    print(X.shape)
    if as_matrix:
        X = X.reshape(
            len(X), config.FLATTEN_MAX_LENGTH, config.FAST_TEXT_EMBEDDING_SIZE
        )
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

INPUT_SIZE = 38400


class RNN(nn.Module):
    def __init__(
        self, input_size, hidden_size=300, n_layers=1, output_size=3, device="cuda"
    ):
        super(RNN, self).__init__()
        self.hidden_dim = hidden_size
        self.n_layers = n_layers
        self.device = device
        self.output_size = output_size

        self.rnn = nn.RNN(
            input_size, hidden_size, n_layers, batch_first=False, dtype=torch.bfloat16
        )
        self.linear = nn.Linear(hidden_size, output_size, dtype=torch.bfloat16)

    def forward(self, x):
        # Initializing hidden state for first input using method defined below
        hidden = self.init_hidden_unbatched()
        out, hidden = self.rnn(x, hidden)
        out = out.contiguous().view(-1, self.hidden_dim)
        out = self.linear(out)
        return out[-1]

    def init_hidden_unbatched(self):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        hidden = torch.zeros(
            self.n_layers, self.hidden_dim, device=self.device, dtype=torch.bfloat16
        )
        return hidden

    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        hidden = torch.zeros(
            batch_size,
            self.n_layers,
            self.hidden_dim,
            device=self.device,
            dtype=torch.bfloat16,
        )
        return hidden


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.dropout_prob = 0.2
        self.linear1 = nn.Linear(INPUT_SIZE, 200, dtype=torch.bfloat16)
        self.linear4 = nn.Linear(200, 3, dtype=torch.bfloat16)

    def forward(self, x):
        a = torch.relu(self.linear1(x))
        b = torch.dropout(a, self.dropout_prob, train=True)
        c = torch.relu(self.linear4(a))
        return c


class Conv(nn.Module):
    def __init__(self):
        super(Conv, self).__init__()
        self.linear1 = nn.Conv2d(8, 200, dtype=torch.bfloat16)
        self.linear4 = nn.Linear(200, 3, dtype=torch.bfloat16)

    def forward(self, x):
        a = torch.relu(self.linear1(x))
        b = torch.dropout(a, self.dropout_prob, train=True)
        c = torch.relu(self.linear4(a))
        return c


print("Using torch device", device)
print("Moving model to device...")
# model = MLP()
# model = RNN(
#     input_size=config.FAST_TEXT_EMBEDDING_SIZE,
#     n_layers=3,
#     output_size=3,
#     hidden_size=100,
# )
model = Conv()
model.to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
criterion = nn.CrossEntropyLoss()
print("Loading data...")
category = "Claim"
feature_column = "flatten_text_matrix"
as_matrix = type(model) != MLP

X_train, y_train = load_dataset(partition="train")
X_train_tensor, y_train_tensor = create_tensors(
    X_train, y_train, use_bfloat16=True, as_matrix=as_matrix
)
X_test, y_test = load_dataset(partition="test")
X_test_tensor, y_test_tensor = create_tensors(
    X_test, y_test, use_bfloat16=True, as_matrix=as_matrix
)

X_val, y_val = load_dataset(partition="val")
X_val_tensor, y_val_tensor = create_tensors(
    X_val, y_val, use_bfloat16=True, as_matrix=as_matrix
)
print("Training...")
## y in math
total_loss = 0.0
j = 0
n_epochs = 100
evaluate_on_epoch_end = True
early_stop = False
should_stop = False
previous_test_loss = 100.00
for epoch in range(n_epochs):
    if should_stop:
        print("Stopping early...")
        break
    model.train()
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
    print("epoch {}, running loss {}".format(epoch, total_loss / j))
    if evaluate_on_epoch_end:
        model.eval()
        with torch.no_grad():
            for partition, X_data, y_data in [
                ("train", X_train_tensor, y_train_tensor),
                ("test", X_test_tensor, y_test_tensor),
                ("val", X_val_tensor, y_val_tensor),
            ]:
                test_total_loss = 0.0
                test_j = 0
                for idx in range(len(X_data)):
                    X = X_data[idx].to(device)
                    y = y_data[idx].to(device)
                    result = model(X)
                    loss = criterion(result, y)
                    test_total_loss += loss.item()
                    test_j += 1
                average_loss = test_total_loss / test_j
                print(f"epoch {epoch}, {partition} loss {average_loss}")

                if partition == "test":
                    test_loss_diff = previous_test_loss - average_loss
                    print(f"test loss diff: {test_loss_diff}")
                    if early_stop and test_loss_diff < -0.01:
                        should_stop = True
                    previous_test_loss = average_loss

print("Cleaning training data...")
del X_train
del y_train
del X_train_tensor
del y_train_tensor


def test_model_on(partition, as_matrix):
    print("Testing on partition... ", partition)
    X, y = load_dataset(partition)
    X_tensor, y_tensor = create_tensors(X, y, use_bfloat16=True, as_matrix=as_matrix)

    hits = 0.0
    misses = 0.0
    total_loss = 0.0
    j = 0
    model.eval()
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


test_model_on("train", as_matrix=as_matrix)
test_model_on("test", as_matrix=as_matrix)
test_model_on("val", as_matrix=as_matrix)
