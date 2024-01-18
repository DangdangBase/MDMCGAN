import argparse
import numpy as np
import csv
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import f1_score, accuracy_score


import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np


from models.mdmcgan import Generator as mdmcgan_gen
from models.cond_wgan_gp import Generator as cond_wgan_gp_gen


parser = argparse.ArgumentParser()
parser.add_argument(
    "--n_epochs", type=int, default=500, help="number of epochs of training"
)
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.00005, help="adam: learning rate")
parser.add_argument(
    "--b1",
    type=float,
    default=0.5,
    help="adam: decay of first order momentum of gradient",
)
parser.add_argument(
    "--b2",
    type=float,
    default=0.999,
    help="adam: decay of first order momentum of gradient",
)
parser.add_argument(
    "--n_cpu",
    type=int,
    default=8,
    help="number of cpu threads to use during batch generation",
)
parser.add_argument(
    "--latent_dim", type=int, default=100, help="dimensionality of the latent space"
)
parser.add_argument(
    "--feature_size", type=int, default=128, help="size of each feature dimension"
)
parser.add_argument("--feature_num", type=int, default=3, help="number of features")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument(
    "--n_critic",
    type=int,
    default=5,
    help="number of training steps for discriminator per iter",
)
parser.add_argument(
    "--clip_value",
    type=float,
    default=0.01,
    help="lower and upper clip value for disc. weights",
)
parser.add_argument(
    "--sample_interval", type=int, default=400, help="interval betwen image samples"
)
parser.add_argument(
    "--dataset",
    type=str,
    choices=["mnist", "fashion", "mnist_c"],
    default="uci_har",
    help="dataset to use",
)
parser.add_argument(
    "--num_modalities",
    type=int,
    default=2,
    help="number of modalities",
)
parser.add_argument(
    "--non_iid",
    action="store_true",
    help="use non-iid dataset",
)
parser.add_argument(
    "--no-non_iid",
    dest="non_iid",
    action="store_false",
    help="use iid dataset",
)
parser.add_argument(
    "--algorithm",
    type=str,
    choices=["mdmcgan", "cond_wgan_gp", "orig_cond_wgan_gp"],
    default="mdmcgan",
    help="feature generating algorithm to use",
)
parser.set_defaults(non_iid=True)

parser.add_argument(
    "--avg_interval", type=int, default=4, help="interval of graph smoothing"
)
opt = parser.parse_args()

opt.feature_shape = (opt.channels, opt.feature_size, opt.feature_num)
opt.n_classes = 6

print(opt)

cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda" if cuda else "cpu")

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

generator = None
generators = []


processed_folder = "UCI_HAR/processed_data"
filtered_folder = "UCI_HAR/filtered_data"

labels = np.load(f"{processed_folder}/labels.npy")
acc_features = np.load(f"{processed_folder}/acc_features.npy")
gyro_features = np.load(f"{processed_folder}/gyro_features.npy")

acc_idx = np.where((labels >= 0) & (labels <= 2))[0]
delete_acc_idx = np.random.choice(acc_idx, size=5 * len(acc_idx) // 6, replace=False)
delete_acc_y = labels[delete_acc_idx]
gyro_for_deleted_acc = gyro_features[delete_acc_idx]


gyro_idx = np.where((labels >= 3) & (labels <= 5))[0]
delete_gyro_idx = np.random.choice(gyro_idx, size=5 * len(gyro_idx) // 6, replace=False)
delete_gyro_y = labels[delete_gyro_idx]
acc_for_deleted_gyro = acc_features[delete_gyro_idx]


removed_idx = np.unique(np.concatenate((delete_acc_idx, delete_gyro_idx), 0))
remain_y = np.delete(labels, removed_idx, axis=0)
remain_acc = np.delete(acc_features, removed_idx, axis=0)
remain_gyro = np.delete(gyro_features, removed_idx, axis=0)

remain_x = np.concatenate([remain_acc, remain_gyro], axis=3)
remain_x = np.squeeze(remain_x)
remain_x = np.concatenate(np.moveaxis(remain_x, 2, 0), axis=1)


X_train, X_test, Y_train, Y_test = train_test_split(
    remain_x, remain_y, test_size=0.2, random_state=0
)


def get_generator(non_iid, algorithm):
    if algorithm == "orig_cond_wgan_gp":
        cur_generators = []
        for i in range(opt.num_modalities):
            generators.append(cond_wgan_gp_gen(opt))
            generators[i].load_state_dict(
                torch.load(
                    f"generator/{non_iid}_{algorithm}_{i}",
                    map_location=torch.device("cpu"),
                )
            )

        generators = cur_generators
    else:
        cur_generator = (
            mdmcgan_gen(opt) if algorithm == "mdmcgan" else cond_wgan_gp_gen(opt)
        )
        cur_generator.load_state_dict(
            torch.load(
                f"generator/{non_iid}_{algorithm}",
                map_location=torch.device("cpu"),
            )
        )
        generator = cur_generator


def interval_avg(data: np.ndarray, interval=4):
    # 8개 datapoint로 평균
    if data.ndim == 3:
        data = data.reshape((data.shape[0], data.shape[1]))
    new_points = data.shape[1] // interval
    if data.shape[1] % interval > 0:
        new_points += 1
    new_data = np.zeros((data.shape[0], new_points))
    for i in range(new_points):
        if i < new_points - 1:
            new_data[:, i] = data[:, i * interval : (i + 1) * interval].mean(axis=1)
        else:
            new_data[:, i] = data[:, i * interval :].mean(axis=1)

    return new_data


def draw_feature_plot():
    if opt.algorithm == "orig_cond_wgan_gp":
        generators = []
        for i in range(opt.num_modalities):
            generators.append(cond_wgan_gp_gen(opt))
            generators[i].load_state_dict(
                torch.load(
                    f"generator/{'non_iid' if opt.non_iid else 'iid'}_{opt.algorithm}_{i}",
                    map_location=torch.device("cpu"),
                )
            )
        else:
            generator = (
                mdmcgan_gen(opt)
                if opt.algorithm == "mdmcgan"
                else cond_wgan_gp_gen(opt)
            )
            generator.load_state_dict(
                torch.load(
                    f"generator/{'non_iid' if opt.non_iid else 'iid'}_{opt.algorithm}",
                    map_location=torch.device("cpu"),
                )
            )

    cur_gen_label = 2

    labels = torch.tensor([cur_gen_label for _ in range(10)])
    gen_features = gen_fake_features(opt.algorithm, labels)

    tmp = np.mean(gen_features.numpy(), axis=0)
    tmp = tmp.swapaxes(0, 1)
    tmp = interval_avg(tmp, opt.avg_interval)

    for i in range(6):
        plt.subplot(6, 2, i + 1)
        plt.plot(tmp[i])
        plt.title(f"gen_{i}")

    processed_folder = "UCI_HAR/processed_data"

    labels = np.load(f"{processed_folder}/labels.npy")
    acc_features = np.load(f"{processed_folder}/acc_features.npy")
    gyro_features = np.load(f"{processed_folder}/gyro_features.npy")

    gen_label_idx = np.where(labels == cur_gen_label)[0]
    # print(gen_label_idx)
    # exit()

    labels_filtered = labels[gen_label_idx]
    acc_features = acc_features[gen_label_idx]
    gyro_features = gyro_features[gen_label_idx]

    data = np.concatenate([acc_features, gyro_features], axis=3)
    data = np.squeeze(data)
    data = np.mean(data, axis=0)
    tmp = np.hsplit(data, 6)
    tmp = np.array(tmp)
    tmp = interval_avg(tmp, opt.avg_interval)

    for i in range(6):
        plt.subplot(6, 2, 6 + i + 1)
        plt.plot(tmp[i])
        plt.title(f"real_{i}")

    plt.show()


def gen_fake_features(al, labels):
    z = Tensor(np.random.normal(0, 1, (len(labels), opt.latent_dim)))

    with torch.no_grad():
        if al == "mdmcgan":
            modals = []
            for i in range(opt.num_modalities):
                modals.append(torch.full((len(labels),), i))

            gen_features_list = []
            for cur_modal in modals:
                gen_features_list.append(generator(z, labels, cur_modal))

            gen_features = torch.cat(gen_features_list, dim=3)

        elif al == "cond_wgan_gp":
            gen_features = generator(z, labels)
            gen_copy = gen_features.clone().detach()
            gen_features = torch.cat([gen_features, gen_copy], dim=3)
        else:
            gen_features_list = []
            for i in range(opt.num_modalities):
                gen_features_list.append(generators[i](z, labels))
            gen_features = torch.cat(gen_features_list, dim=3)
    return gen_features


def gen_blended_features(al, non_iid, X_train, Y_train):
    gen_acc = gen_fake_features(al, torch.from_numpy(delete_acc_y))
    gen_acc = np.split(gen_acc.numpy(), 2, axis=3)[0]

    gen_gyro = gen_fake_features(al, torch.from_numpy(delete_gyro_y))
    gen_gyro = np.split(gen_gyro.numpy(), 2, axis=3)[1]

    gen_y = np.concatenate([delete_acc_y, delete_gyro_y], axis=0)

    gen_x = np.concatenate(
        [
            np.concatenate([gen_acc, gyro_for_deleted_acc], axis=3),
            np.concatenate([acc_for_deleted_gyro, gen_gyro], axis=3),
        ],
        axis=0,
    )
    gen_x = np.squeeze(gen_x)
    gen_x = np.concatenate(np.moveaxis(gen_x, 2, 0), axis=1)

    # if non_iid == "iid":
    gen_labels = [i for i in range(6) for _ in range(500)]
    gen_data = gen_fake_features(al, torch.tensor(gen_labels))

    gen_data = np.squeeze(gen_data.numpy())
    gen_data = np.concatenate(np.moveaxis(gen_data, 2, 0), axis=1)

    gen_x = np.concatenate([gen_x, gen_data], axis=0)
    gen_y = np.concatenate([gen_y, gen_labels], axis=0)

    X_train = np.concatenate([X_train, gen_x], axis=0)
    Y_train = np.concatenate([Y_train, gen_y], axis=0)

    shuffler = np.random.permutation(len(X_train))
    X_train = X_train[shuffler]
    Y_train = Y_train[shuffler]

    return X_train, Y_train


# def evaluate_all_algorithms():

params = {
    "max_depth": [200, 250, 300],
    "n_estimators": [200, 250],
    "min_samples_leaf": [2],
    "min_samples_split": [2],
}

scoring = ["f1_weighted", "accuracy", "f1_macro"]

result_f = open(
    f"score_{''.join([str(i) for i in params['max_depth']])}_result.csv", "w"
)
writer = csv.writer(result_f)
writer.writerow(["algorithm", "f1_weighted", "accuracy", "f1_macro"])


# al = "mdmcgan"
# non_iid = "iid"
# # non_iid = "non_iid"

# generator = mdmcgan_gen(opt)
# generator.load_state_dict(
#     torch.load(
#         f"generator/{non_iid}_{al}",
#         map_location=torch.device("cpu"),
#     )
# )

# new_train_x, new_train_y = gen_blended_features(al, non_iid, X_train, Y_train)


# clf = RandomForestClassifier(random_state=0, n_jobs=-1)

# # params = {
# #     "n_neighbors": [5, 7, 10],
# #     "weights": ["uniform", "distance"],
# #     "metric": ["euclidean", "manhattan", "minkowski"],
# # }
# # knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
# # clf = RandomForestClassifier(random_state=0, n_jobs=-1)

# grid_cv = GridSearchCV(
#     clf,
#     param_grid=params,
#     cv=6,
#     n_jobs=-1,
#     scoring=scoring,
#     refit="accuracy",
# )
# grid_cv.fit(new_train_x, new_train_y)

# # predicted = grid.predict(X_test)
# predicted = grid_cv.predict(X_test)


# f1_weighted = round(f1_score(Y_test, predicted, average="weighted"), 4)
# f1_macro = round(f1_score(Y_test, predicted, average="macro"), 4)
# accuracy = round(accuracy_score(Y_test, predicted), 4)

# print(f"{f1_weighted:.4f} {f1_macro:.4f} {accuracy:.4f}")
# print(grid_cv.score(X_test, Y_test))
# print(grid_cv.best_params_)


for non_iid in ["iid", "non_iid"]:
    for al in ["mdmcgan", "cond_wgan_gp", "orig_cond_wgan_gp"]:
        # for non_iid in ["iid", "non_iid"]:
        #     for al in ["mdmcgan"]:
        if al == "orig_cond_wgan_gp":
            # generators = []
            for i in range(opt.num_modalities):
                generators.append(cond_wgan_gp_gen(opt))
                generators[i].load_state_dict(
                    torch.load(
                        f"generator/{non_iid}_{al}_{i}",
                        map_location=torch.device("cpu"),
                    )
                )
        else:
            generator = mdmcgan_gen(opt) if al == "mdmcgan" else cond_wgan_gp_gen(opt)
            generator.load_state_dict(
                torch.load(
                    f"generator/{non_iid}_{al}",
                    map_location=torch.device("cpu"),
                )
            )

        print(f"{non_iid}_{al}")

        new_train_x, new_train_y = gen_blended_features(al, non_iid, X_train, Y_train)

        rf_clf = RandomForestClassifier(random_state=0, n_jobs=-1)
        grid_cv = GridSearchCV(
            rf_clf,
            param_grid=params,
            cv=6,
            n_jobs=-1,
            scoring=scoring,
            refit="accuracy",
        )
        grid_cv.fit(new_train_x, new_train_y)

        # predicted = grid.predict(X_test)
        predicted = grid_cv.predict(X_test)

        f1_weighted = round(f1_score(Y_test, predicted, average="weighted"), 4)
        f1_macro = round(f1_score(Y_test, predicted, average="macro"), 4)
        accuracy = round(accuracy_score(Y_test, predicted), 4)

        # tmp = [f"{non_iid}_{al}"]
        for score in scoring:
            print(
                f"{grid_cv.cv_results_['mean_test_' + score][grid_cv.best_index_]:.4f}"
            )

        print(f"{f1_weighted:.4f} {f1_macro:.4f} {accuracy:.4f}")
        print(grid_cv.score(X_test, Y_test))
        print(grid_cv.best_params_)

        writer.writerow([f"{non_iid}_{al}", f1_weighted, f1_macro, accuracy])
        result_f.flush()


# draw_feature_plot()
# evaluate_all_algorithms()
