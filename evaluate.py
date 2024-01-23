import numpy as np
import torch
import os
import csv

import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, RocCurveDisplay

from models.mdmcgan import Generator as mdmcgan_gen
from models.cond_wgan_gp import Generator as cond_wgan_gp_gen
from train_params import opt


os.makedirs("plots", exist_ok=True)

cuda = False
device = torch.device("cuda" if cuda else "cpu")

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

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


params = {
    "max_depth": [50],
    "n_estimators": [300],
    "min_samples_leaf": [8],
    "min_samples_split": [8],
}

algorithms = ["mdmcgan", "orig_cond_wgan_gp", "cond_wgan_gp"]
scoring = ["f1_weighted", "accuracy", "f1_macro"]

result_f = open(f"score_{opt.remove_labels_num}_{opt.filter_ratio}_result.csv", "w")
writer = csv.writer(result_f)
writer.writerow(["algorithm", "f1_weighted", "accuracy", "f1_macro"])


clf_list = [
    RandomForestClassifier(
        max_depth=50,
        n_estimators=300,
        min_samples_leaf=8,
        min_samples_split=8,
        random_state=0,
        n_jobs=-1,
    )
    for _ in range(3)
]


for non_iid_str in ["non_iid"]:
    for i, algorithm in enumerate(algorithms):
        generator = None
        generators = []

        if algorithm == "orig_cond_wgan_gp":
            generators = get_generator(algorithm, non_iid_str)
        else:
            generator = get_generator(algorithm, non_iid_str)

        print(f"{non_iid_str}_{algorithm}")

        X_train, Y_train, X_test, Y_test = gen_blended_features(algorithm, non_iid_str)

        clf_list[i].fit(X_train, Y_train)

        predicted = clf_list[i].predict(X_test)

        f1_weighted = round(f1_score(Y_test, predicted, average="weighted"), 4)
        f1_macro = round(f1_score(Y_test, predicted, average="macro"), 4)
        accuracy = round(accuracy_score(Y_test, predicted), 4)

        print(f"{f1_weighted}, {f1_macro}, {accuracy}")

        writer.writerow([f"{non_iid_str}_{algorithm}", f1_weighted, f1_macro, accuracy])
        result_f.flush()

result_f.close()


Y_test_onehot = np.eye(6)[Y_test].astype(int)
colors = ["blue", "darkorange", "teal"]

for j in range(6):
    fig = plt.figure(figsize=(8, 8))
    fig.set_facecolor("white")
    ax = fig.add_subplot()
    ax.set_title(f"label {j} vs rest")

    for i, rf_clf in enumerate(clf_list):
        predicted_prob = clf_list[i].predict_proba(X_test)

        RocCurveDisplay.from_predictions(
            Y_test_onehot[:, j],
            predicted_prob[:, j],
            name=algorithms[i],
            color=colors[i],
            ax=ax,
        )

    ax.plot([0, 1], [0, 1], color="red", label="Chance Level")
    ax.legend()

    plt.savefig(f"plots/label_{j}.png")
