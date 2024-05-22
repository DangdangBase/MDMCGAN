import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import csv

from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import DataLoader, TensorDataset, random_split

from models.mdmcgan import Generator as mdmcgan_gen
from models.cond_wgan_gp import Generator as cond_wgan_gp_gen
from train_params import opt

os.makedirs("plots", exist_ok=True)
debug = True
cuda = True
device = torch.device("cuda" if cuda else "cpu")

iid_labels = 1500
non_iid_labels = 1500
ratio = 18

def flatten_features(np_arr):
    np_arr = np.squeeze(np_arr)
    np_arr = np.concatenate(np.moveaxis(np_arr, 2, 0), axis=1)
    return np_arr

def get_generator(algorithm, non_iid):
    if algorithm == "orig_cond_wgan_gp":
        cur_generators = []
        for i in range(opt.num_modalities):
            cur_generators.append(cond_wgan_gp_gen(opt))
            cur_generators[i].load_state_dict(
                torch.load(
                    f"generator/{non_iid}_{opt.remove_labels_num}_{opt.filter_ratio}_{algorithm}_{i}",
                    map_location=torch.device(device),
                )
            )
        return cur_generators
    else:
        cur_generator = (
            mdmcgan_gen(opt) if algorithm == "mdmcgan" else cond_wgan_gp_gen(opt)
        )
        cur_generator.load_state_dict(
            torch.load(
                f"generator/{non_iid}_{opt.remove_labels_num}_{opt.filter_ratio}_{algorithm}",
                map_location=torch.device("cpu"),
            )
        )
        return cur_generator

def gen_fake_features(algorithm, labels):
    z = torch.randn(len(labels), opt.latent_dim, device=device)
    labels = labels.to(device)  # Ensure labels are on the correct device

    with torch.no_grad():
        if algorithm == "mdmcgan":
            modals = []
            for i in range(opt.num_modalities):
                modals.append(torch.full((len(labels),), i, device=device))

            gen_features_list = []
            for cur_modal in modals:
                # Ensure generator is on the correct device (if not done during loading)
                generator.to(device)
                gen_features_list.append(generator(z, labels, cur_modal))

            gen_features = torch.cat(gen_features_list, dim=3)

        elif algorithm == "cond_wgan_gp":
            # Ensure generator is on the correct device (if not done during loading)
            generator.to(device)
            gen_features = generator(z, labels)
            gen_copy = gen_features.clone().detach()
            gen_features = torch.cat([gen_features, gen_copy], dim=3)
        else:
            gen_features_list = []
            for i in range(opt.num_modalities):
                # Ensure each generator in the list is on the correct device
                generators[i].to(device)
                gen_features_list.append(generators[i](z, labels))
            gen_features = torch.cat(gen_features_list, dim=3)
    return gen_features


def gen_blended_features(algorithm, non_iid_str):
    uci_har_folder = "UCI_HAR/filtered_data"

    if non_iid_str == "non_iid":
        data = np.load(
            f"{uci_har_folder}/{non_iid_str}_{opt.remove_labels_num}_{opt.filter_ratio}.npz"
        )
    else:
        data = np.load(f"{uci_har_folder}/{non_iid_str}.npz")

    X_train = flatten_features(data["X_train"])
    X_test = flatten_features(data["X_test"])
    Y_train = data["Y_train"]
    Y_test = data["Y_test"]

    if non_iid_str == "non_iid":
        y_remain_gyro = data["y_remain_gyro"]
        gyro_remain = data["gyro_remain"]
        y_remain_acc = data["y_remain_acc"]
        acc_remain = data["acc_remain"]

    data.close()

    if non_iid_str == "non_iid":
        gen_acc = gen_fake_features(algorithm, torch.from_numpy(y_remain_gyro).to(device))
        gen_acc = np.split(gen_acc.cpu().numpy(), 2, axis=3)[0]

        gen_gyro = gen_fake_features(algorithm, torch.from_numpy(y_remain_acc).to(device))
        gen_gyro = np.split(gen_gyro.cpu().numpy(), 2, axis=3)[1]

        gen_y = np.concatenate([y_remain_gyro, y_remain_acc], axis=0)

        gen_x = np.concatenate(
            [
                np.concatenate([gen_acc, gyro_remain], axis=3),
                np.concatenate([acc_remain, gen_gyro], axis=3),
            ],
            axis=0,
        )
        gen_x = flatten_features(gen_x)

    gen_labels = [
        i
        for i in range(6)
        for _ in range(non_iid_labels if non_iid_str == "non_iid" else iid_labels)
    ]

    gen_data = gen_fake_features(algorithm, torch.tensor(gen_labels).to(device))
    gen_data = flatten_features(gen_data.cpu().numpy())

    if non_iid_str == "iid":
        gen_x = gen_data
        gen_y = gen_labels

    # X_train = np.concatenate([X_train, gen_x, gen_data], axis=0)
    # Y_train = np.concatenate([Y_train, gen_y, gen_labels], axis=0)

# For the case of debug
    if debug:
        X_train = np.concatenate([X_train], axis=0)
        Y_train = np.concatenate([Y_train], axis=0)

    shuffler = np.random.permutation(len(X_train))
    X_train = X_train[shuffler]
    Y_train = Y_train[shuffler]

    return X_train, Y_train, X_test, Y_test

import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size=6, hidden_size=32, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=32, hidden_size=32, batch_first=True)
        self.conv1 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=2)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.batch_norm = nn.BatchNorm1d(128)
        self.fc = nn.Linear(128, 6)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # print(f"Input shape: {x.shape}")
        x, _ = self.lstm1(x)
        # print(f"After LSTM1: {x.shape}")
        x, _ = self.lstm2(x)
        # print(f"After LSTM2: {x.shape}")
        # Ensure x has three dimensions before permuting
        if x.ndim != 3:
            raise RuntimeError("Expected 3 dimensions but got " + str(x.ndim))
        x = x.permute(0, 2, 1)  # Correct permute call to swap the last two dimensions
        # print(f"After Permute: {x.shape}")
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.global_avg_pool(x)
        x = x.squeeze(-1)
        x = self.batch_norm(x)
        x = self.fc(x)
        x = self.softmax(x)
        # print(f"After Softmax: {x.shape}")
        return x





# Hyperparameters
base_lr = 0.001
base_batch_size = 32
batch_size = 64
epoch = 15
params = {
    "epochs": epoch,
    "batch_size": batch_size,
    "learning_rate": base_lr * (batch_size / base_batch_size),
    "dropout_rate": 0.5,
    "use_batch_norm": True,
}

algorithms = ["mdmcgan", "orig_cond_wgan_gp", "cond_wgan_gp"]
scoring = ["f1_weighted", "accuracy", "f1_macro"]

result_f = open(f"score_{opt.remove_labels_num}_{opt.filter_ratio}_result.csv", "w")
writer = csv.writer(result_f)
writer.writerow(["algorithm", "f1_weighted", "f1_macro", "accuracy"])

# Placeholder for generators list
generators = [None, None, None]

for non_iid_str in ["non_iid"]:
    for i, algorithm in enumerate(algorithms):
        generator = None

        if algorithm == "orig_cond_wgan_gp":
            generators = get_generator(algorithm, non_iid_str)
        else:
            generator = get_generator(algorithm, non_iid_str)

        print(f"{non_iid_str}_{algorithm}")

        X_train, Y_train, X_test, Y_test = gen_blended_features(algorithm, non_iid_str)

        # One-hot encode the labels for training the LSTM
        n_classes = 6  # Adjust this based on the actual number of classes you have
        Y_train_onehot = torch.eye(n_classes)[torch.tensor(Y_train, dtype=torch.long)].to(device)
        Y_test_onehot = torch.eye(n_classes)[torch.tensor(Y_test, dtype=torch.long)].to(device)

        # Define the number of features and timesteps
        n_features, n_timesteps = 6, 128  # Adjust these based on your data shape and LSTM configuration

        # Convert NumPy arrays to PyTorch tensors and reshape for LSTM input
        X_train = torch.tensor(X_train, dtype=torch.float32).reshape(-1, n_timesteps, n_features).to(device)
        X_test = torch.tensor(X_test, dtype=torch.float32).reshape(-1, n_timesteps, n_features).to(device)

        # Create dataset objects for PyTorch
        train_dataset = TensorDataset(X_train, Y_train_onehot)
        test_dataset = TensorDataset(X_test, Y_test_onehot)

        # Splitting the training data into training and validation sets
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

        # Creating data loaders
        batch_size = params["batch_size"]  # Ensure params dictionary has 'batch_size' key
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Define the LSTM model with input shape (might need to define create_lstm_model if not defined)
        input_shape = (n_timesteps, n_features)
        model = LSTMModel().to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=params["learning_rate"])

        for epoch in range(params["epochs"]):
            model.train()
            for X_batch, Y_batch in train_loader:
                optimizer.zero_grad()
                output = model(X_batch)
                loss = criterion(output, Y_batch)
                loss.backward()
                optimizer.step()
            
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for X_batch, Y_batch in val_loader:
                    output = model(X_batch)
                    val_loss += criterion(output, Y_batch).item()
            val_loss /= len(val_loader)
            print(f"Epoch {epoch+1}, Validation Loss: {val_loss}")

        model.eval()
        all_preds = []
        with torch.no_grad():
            for X_batch, _ in test_loader:
                output = model(X_batch)
                preds = torch.argmax(output, dim=1)
                all_preds.append(preds.cpu().numpy())
        all_preds = np.concatenate(all_preds)

        f1_weighted = round(f1_score(Y_test, all_preds, average="weighted"), 4)
        f1_macro = round(f1_score(Y_test, all_preds, average="macro"), 4)
        accuracy = round(accuracy_score(Y_test, all_preds), 4)

        print(f"{f1_weighted}, {f1_macro}, {accuracy}")

        writer.writerow(
            [f"{non_iid_str}_{algorithm}", f1_weighted, f1_macro, accuracy]
        )
        result_f.flush()

result_f.close()
