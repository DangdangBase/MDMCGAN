import os
import shutil
import numpy as np
import pandas as pd


def load_data_UCIHAR(dataset_path, file_list, type):
    x_data_list = []
    for item in file_list:
        item_data = np.array(
            pd.read_csv(
                dataset_path + type + "/Inertial Signals/" + item + type + ".txt",
                delim_whitespace=True,
                header=None,
            )
        )
        x_data_list.append(item_data)
    x = np.stack(x_data_list, -1)

    y = np.array(
        pd.read_csv(dataset_path + type + "/y_" + type + ".txt", names=["Activity"])
    )
    y = y.squeeze()
    return x, y


def preprocess_UCIHAR():
    dataset_path = "UCI_HAR/"

    file_list = [
        "body_acc_x_",
        "body_acc_y_",
        "body_acc_z_",
        "body_gyro_x_",
        "body_gyro_y_",
        "body_gyro_z_",
    ]
    x_train, y_train = load_data_UCIHAR(dataset_path, file_list, "train")
    x_test, y_test = load_data_UCIHAR(dataset_path, file_list, "test")

    x = np.concatenate([x_train, x_test], axis=0)
    y = np.concatenate([y_train, y_test], axis=0)

    lower_bound = np.array(
        [
            -0.7270811,
            -0.8285408496200001,
            -0.72422586782,
            -2.5482600237,
            -2.3043757869,
            -1.698266,
        ]
    )
    upper_bound = np.array(
        [1.072854472100006, 0.620366, 0.6387655, 2.643864, 3.4056461708005163, 1.60952]
    )
    diff = upper_bound - lower_bound

    if os.path.exists(dataset_path + "processed_data/"):
        shutil.rmtree(dataset_path + "processed_data/")
    os.mkdir(dataset_path + "processed_data/")

    x = 2 * (x - lower_bound) / diff - 1

    x[x > 1] = 1.0
    x[x < -1] = -1.0

    y = y - 1

    x = np.expand_dims(x, axis=1)
    splitted = np.split(x, 2, axis=3)
    x_acc = splitted[0]
    x_gyro = splitted[1]

    np.save(dataset_path + "/processed_data/acc_features", x_acc)
    np.save(dataset_path + "/processed_data/gyro_features", x_gyro)
    np.save(dataset_path + "/processed_data/labels", y)


preprocess_UCIHAR()
