import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import csv
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import mixed_precision
from tensorflow.keras.optimizers import Adam

import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, roc_curve, RocCurveDisplay
from torch.utils.data import DataLoader, TensorDataset

from models.mdmcgan import Generator as mdmcgan_gen
from models.cond_wgan_gp import Generator as cond_wgan_gp_gen
from train_params import opt

os.makedirs("plots", exist_ok=True)

mixed_precision.set_global_policy('mixed_float16')

#cuda = torch.cuda.is_available()
#device = torch.device("cuda" if cuda else "cpu")
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

#Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
Tensor = torch.FloatTensor

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
                    map_location=torch.device("cpu"),
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
    z = Tensor(np.random.normal(0, 1, (len(labels), opt.latent_dim)))

    with torch.no_grad():
        if algorithm == "mdmcgan":
            modals = []
            for i in range(opt.num_modalities):
                modals.append(torch.full((len(labels),), i))

            gen_features_list = []
            for cur_modal in modals:
                gen_features_list.append(generator(z, labels, cur_modal))

            gen_features = torch.cat(gen_features_list, dim=3)

        elif algorithm == "cond_wgan_gp":
            gen_features = generator(z, labels)
            gen_copy = gen_features.clone().detach()
            gen_features = torch.cat([gen_features, gen_copy], dim=3)
        else:
            gen_features_list = []
            for i in range(opt.num_modalities):
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
        gen_acc = gen_fake_features(algorithm, torch.from_numpy(y_remain_gyro))
        gen_acc = np.split(gen_acc.numpy(), 2, axis=3)[0]

        gen_gyro = gen_fake_features(algorithm, torch.from_numpy(y_remain_acc))
        gen_gyro = np.split(gen_gyro.numpy(), 2, axis=3)[1]

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

    gen_data = gen_fake_features(algorithm, torch.tensor(gen_labels))
    gen_data = flatten_features(gen_data.numpy())

    if non_iid_str == "iid":
        gen_x = gen_data
        gen_y = gen_labels

    X_train = np.concatenate([X_train, gen_x, gen_data], axis=0)
    Y_train = np.concatenate([Y_train, gen_y, gen_labels], axis=0)

    shuffler = np.random.permutation(len(X_train))
    X_train = X_train[shuffler]
    Y_train = Y_train[shuffler]

    return X_train, Y_train, X_test, Y_test

# Function to create LSTM model
def create_lstm_model(input_shape, params):
    model = Sequential([
        LSTM(params['lstm_units_1'], input_shape=input_shape, return_sequences=True, dropout=params['dropout_rate'], recurrent_dropout=params['recurrent_dropout']),
        Dropout(params['dropout_rate']),
        LSTM(params['lstm_units_2'], dropout=params['dropout_rate'], recurrent_dropout=params['recurrent_dropout']),
        BatchNormalization() if params['use_batch_norm'] else None,
        Dropout(params['dropout_rate']),
        Dense(6, activation=params['activation_function'])  # Assuming 6 classes in Y_train
    ])
    optimizer = Adam(learning_rate=params['learning_rate'])
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Hyperparameters
base_lr = 0.001
base_batch_size = 32
batch_size = 64
epoch = 30
params = {
    "epochs": epoch,
    "batch_size": batch_size,
    "learning_rate": base_lr * (batch_size / base_batch_size),
    "lstm_units_1": 64,
    "lstm_units_2": 64,
    "dropout_rate": 0.5,
    "use_batch_norm": True,
    "activation_function": 'softmax',
    "recurrent_dropout": 0,
    "optimizer_choice": 'adam',
    "learning_rate_decay": 0.99
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
        Y_train_onehot = np.eye(6)[Y_train].astype(int)
        Y_test_onehot = np.eye(6)[Y_test].astype(int)

        # Reshape input data to be 3D [samples, timesteps, features]
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

        input_shape = (X_train.shape[1], X_train.shape[2])

        lstm_model = create_lstm_model(input_shape, params)

        # Train the LSTM model
        early_stopping = EarlyStopping(monitor='val_loss', patience=5)
        lstm_model.fit(X_train, Y_train_onehot, epochs=params["epochs"], batch_size=params["batch_size"], validation_split=0.2, callbacks=[early_stopping])

        predicted_prob = lstm_model.predict(X_test)
        predicted = np.argmax(predicted_prob, axis=1)

        f1_weighted = round(f1_score(Y_test, predicted, average="weighted"), 4)
        f1_macro = round(f1_score(Y_test, predicted, average="macro"), 4)
        accuracy = round(accuracy_score(Y_test, predicted), 4)

        print(f"{f1_weighted}, {f1_macro}, {accuracy}")

        writer.writerow(
            [f"{non_iid_str}_{algorithm}", f1_weighted, f1_macro, accuracy]
        )
        result_f.flush()

result_f.close()

Y_test_onehot = np.eye(6)[Y_test].astype(int)
colors = ["blue", "darkorange", "teal"]

for j in range(6):
    fig = plt.figure(figsize=(8, 8))
    fig.set_facecolor("white")
    ax = fig.add_subplot()
    ax.set_title(f"label {j} vs rest")

    for i, algorithm in enumerate(algorithms):
        predicted_prob = lstm_model.predict(X_test)

        RocCurveDisplay.from_predictions(
            Y_test_onehot[:, j],
            predicted_prob[:, j],
            name=algorithm,
            color=colors[i],
            ax=ax,
        )

    ax.plot([0, 1], [0, 1], color="red", label="Chance Level")
    ax.legend()

    plt.savefig(f"plots/label_{j}.png")
