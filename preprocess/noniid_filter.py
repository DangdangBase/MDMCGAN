import argparse
import numpy as np
from sklearn.model_selection import train_test_split


parser = argparse.ArgumentParser()
parser.add_argument("--non_iid", action="store_true")
parser.add_argument("--no-non_iid", dest="non_iid", action="store_false")
parser.set_defaults(non_iid=True)
parser.add_argument("--remove_labels_num", type=int, default=3, choices=[1, 2, 3])
parser.add_argument("--filter_ratio", type=float, default=0.2)
opt = parser.parse_args()


processed_folder = "UCI_HAR/processed_data"
filtered_folder = "UCI_HAR/filtered_data"


def filter_processed_data(non_iid=True, **kwargs):
    non_iid_str = "non_iid" if non_iid else "iid"

    labels = np.load(f"{processed_folder}/labels.npy")
    acc_features = np.load(f"{processed_folder}/acc_features.npy")
    gyro_features = np.load(f"{processed_folder}/gyro_features.npy")

    if non_iid:
        remove_labels_num = kwargs["remove_labels_num"]
        filter_ratio = kwargs["filter_ratio"]
        assert 1 <= remove_labels_num and remove_labels_num <= 3

        acc_idx = np.where((labels >= 0) & (labels <= (remove_labels_num - 1)))[0]
        acc_idx_del = np.random.choice(
            acc_idx, size=int(len(acc_idx) * filter_ratio), replace=False
        )
        y_remain_gyro = labels[acc_idx_del]
        gyro_remain = gyro_features[acc_idx_del]

        gyro_idx = np.where((labels >= (6 - remove_labels_num)) & (labels <= 5))[0]
        gyro_idx_del = np.random.choice(
            gyro_idx, size=int(len(gyro_idx) * filter_ratio), replace=False
        )
        y_remain_acc = labels[gyro_idx_del]
        acc_remain = acc_features[gyro_idx_del]

        removed_idx = np.unique(np.concatenate((acc_idx_del, gyro_idx_del), 0))
        remain_y = np.delete(labels, removed_idx, axis=0)

        orig_x = np.concatenate([acc_features, gyro_features], axis=3)
        remain_x = np.delete(orig_x, removed_idx, axis=0)

        X_train, X_test, Y_train, Y_test = train_test_split(
            remain_x, remain_y, test_size=0.2, random_state=0
        )

        np.savez(
            f"{filtered_folder}/{non_iid_str}_{remove_labels_num}_{filter_ratio}",
            X_train=X_train,
            X_test=X_test,
            Y_train=Y_train,
            Y_test=Y_test,
            y_remain_gyro=y_remain_gyro,
            gyro_remain=gyro_remain,
            y_remain_acc=y_remain_acc,
            acc_remain=acc_remain,
        )
    else:
        orig_x = np.concatenate([acc_features, gyro_features], axis=3)
        orig_y = labels

        X_train, X_test, Y_train, Y_test = train_test_split(
            orig_x, orig_y, test_size=0.35, random_state=0
        )

        np.savez(
            f"{filtered_folder}/{non_iid_str}",
            X_train=X_train,
            X_test=X_test,
            Y_train=Y_train,
            Y_test=Y_test,
        )


filter_processed_data(
    non_iid=opt.non_iid,
    remove_labels_num=opt.remove_labels_num,
    filter_ratio=opt.filter_ratio,
)
