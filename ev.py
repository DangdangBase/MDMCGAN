import argparse
import numpy as np
import csv
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, accuracy_score
import torch


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

cuda = False
device = torch.device("cuda" if cuda else "cpu")

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor




def get_generator(non_iid, algorithm):
    if algorithm == "orig_cond_wgan_gp":
        cur_generators = []
        for i in range(opt.num_modalities):
            cur_generators.append(cond_wgan_gp_gen(opt))
            cur_generators[i].load_state_dict(
                torch.load(
                    f"generator/{non_iid}_{algorithm}_{i}",
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
                f"generator/{non_iid}_{algorithm}",
                map_location=torch.device("cpu"),
            )
        )
        return cur_generator

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
    if non_iid == "non_iid":
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

    if non_iid == "iid":
        gen_labels = [i for i in range(6) for _ in range(300)]
    else:
        gen_labels = [i for i in range(6) for _ in range(200)]
    gen_data = gen_fake_features(al, torch.tensor(gen_labels))

    gen_data = np.squeeze(gen_data.numpy())
    gen_data = np.concatenate(np.moveaxis(gen_data, 2, 0), axis=1)

    if non_iid == "iid":
        gen_x = gen_data
        gen_y = gen_labels
    else:
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
    "max_depth": [10, 20, 30],
    "n_estimators": [200, 225, 250],
    "min_samples_leaf": [2],
    "min_samples_split": [2],
}

scoring = ["f1_weighted", "accuracy", "f1_macro"]

result_f = open(
    f"score_{''.join([str(i) for i in params['max_depth']])}_result.csv", "w"
)
writer = csv.writer(result_f)
writer.writerow(["algorithm", "f1_weighted", "accuracy", "f1_macro"])


for non_iid in ["iid", "non_iid"]:
    
    for al in ["mdmcgan", "cond_wgan_gp", "orig_cond_wgan_gp"]:
        generator = None
        generators = []

        processed_folder = "UCI_HAR/processed_data"
        filtered_folder = "UCI_HAR/filtered_data"

        labels = np.load(f"{processed_folder}/labels.npy")
        acc_features = np.load(f"{processed_folder}/acc_features.npy")
        gyro_features = np.load(f"{processed_folder}/gyro_features.npy")

        if non_iid == "non_iid":
            acc_idx = np.where((labels >= 0) & (labels <= 2))[0]
            delete_acc_idx = np.random.choice(acc_idx, size=19 * len(acc_idx) // 20, replace=False)
            delete_acc_y = labels[delete_acc_idx]
            gyro_for_deleted_acc = gyro_features[delete_acc_idx]

            gyro_idx = np.where((labels >= 3) & (labels <= 5))[0]
            delete_gyro_idx = np.random.choice(gyro_idx, size=19 * len(gyro_idx) // 20, replace=False)
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
        if non_iid == "iid":
            X = np.concatenate([acc_features, gyro_features], axis = 3)
            X = np.squeeze(X)
            X = np.concatenate(np.moveaxis(X, 2, 0), axis=1)

            Y = labels
            X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.2, random_state=0
        )
        if al == "orig_cond_wgan_gp":
            generators = get_generator(non_iid, al)
        else:
            generator = get_generator(non_iid, al)

        print(f"{non_iid}_{al}")

        rf_clf = RandomForestClassifier(random_state=0, n_jobs=-1)
        grid_cv = GridSearchCV(
            rf_clf,
            param_grid=params,
            cv=5,
            n_jobs=-1,
            scoring=scoring,
            refit="accuracy",
        )
        
        new_train_x, new_train_y = gen_blended_features(al, non_iid, X_train, Y_train)
        grid_cv.fit(new_train_x, new_train_y)

        predicted = grid_cv.predict(X_test)

        f1_weighted = round(f1_score(Y_test, predicted, average="weighted"), 4)
        f1_macro = round(f1_score(Y_test, predicted, average="macro"), 4)
        accuracy = round(accuracy_score(Y_test, predicted), 4)
        
        print(f"{f1_weighted:.4f} {f1_macro:.4f} {accuracy:.4f}")

        writer.writerow([f"{non_iid}_{al}", f1_weighted, f1_macro, accuracy])
        result_f.flush()