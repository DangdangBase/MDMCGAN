import argparse
import numpy as np
import torch
import csv
import os
import matplotlib.pyplot as plt

from models.mdmcgan import Generator as mdmcgan_gen
from models.cond_wgan_gp import Generator as cond_wgan_gp_gen

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier


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
opt = parser.parse_args()

opt.feature_shape = (opt.channels, opt.feature_size, opt.feature_num)
opt.n_classes = 6

print(opt)

cuda = True if torch.cuda.is_available() else False
mps = True if torch.backends.mps.is_available() else False

if cuda:
    device = "cuda"
elif mps:
    device = "mps"
else:
    device = "cpu"


Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


#############
# evaluate
#############

desired_ratios = [1, 0, 0, 0, 0, 0]
total_samples = 6
counts = [int(ratio * total_samples) for ratio in desired_ratios]


filtered_folder = "./UCI_HAR/filtered_data"
generated_folder = "./UCI_HAR/gen_data"

y = np.load(f"{filtered_folder}/labels.npy")

x_acc = np.load(f"{filtered_folder}/acc_features.npy")
x_acc = np.squeeze(x_acc)
x_acc = np.concatenate(np.moveaxis(x_acc, 2, 0), axis=1)

x_gyro = np.load(f"{filtered_folder}/gyro_features.npy")
x_gyro = np.squeeze(x_gyro)
x_gyro = np.concatenate(np.moveaxis(x_gyro, 2, 0), axis=1)

x = np.concatenate([x_acc, x_gyro], axis=1)

g_X_train, g_X_test, g_Y_train, g_Y_test = train_test_split(
    x, y, test_size=0.1, random_state=0
)


def generate_fake_features():
    with torch.no_grad():
        z = Tensor(np.random.normal(0, 1, (total_samples, opt.latent_dim)))
        labels = torch.cat([torch.full((count,), i) for i, count in enumerate(counts)])

        if opt.algorithm == "mdmcgan":
            modals = []
            for i in range(opt.num_modalities):
                modals.append(torch.full((total_samples,), i))

            gen_features_list = []

            for cur_modal in modals:
                gen_features_list.append(generator(z, labels, cur_modal))

            gen_features = torch.cat(gen_features_list, dim=3)

        elif opt.algorithm == "cond_wgan_gp":
            gen_features = generator(z, labels)
            gen_copy = gen_features.clone().detach()
            gen_features = torch.cat([gen_features, gen_copy], dim=3)
        else:
            gen_features_list = []
            for i in range(opt.num_modalities):
                gen_features_list.append(generators[i](z, labels))
            gen_features = torch.cat(gen_features_list, dim=3)
    return gen_features, labels


def evaluate_fake_features(X_gen, Y_gen):
    X_gen = np.squeeze(X_gen.numpy())
    X_gen = np.concatenate(np.moveaxis(X_gen, 2, 0), axis=1)

    X_train = np.concatenate((g_X_train, X_gen), axis=0)
    Y_train = np.concatenate((g_Y_train, Y_gen), axis=0)

    random_idx = np.random.permutation(len(X_train))
    X_train = X_train[random_idx]
    Y_train = Y_train[random_idx]

    params = {
        "max_depth": [6, 8, 10],
        "n_estimators": [50, 100, 200],
        "min_samples_leaf": [8, 12],
        "min_samples_split": [8, 12],
    }

    rf_clf = RandomForestClassifier(random_state=0, n_jobs=-1)
    grid_cv = GridSearchCV(
        rf_clf, param_grid=params, cv=2, n_jobs=-1, scoring="f1_macro"
    )
    grid_cv.fit(X_train, Y_train)

    f1_score = grid_cv.score(g_X_test, g_Y_test)
    print(f1_score)

    return f1_score


result_f = open(f"{'non_iid' if opt.non_iid else 'iid'}_{opt.algorithm}_score.csv", "w")
writer = csv.writer(result_f)
writer.writerow(["Batch", "F1-score"])

i = 0
while True:
    print(i)

    generator_file_name = (
        f"generator/{'non_iid' if opt.non_iid else 'iid'}_{opt.algorithm}_{i}"
    )

    if not os.path.isfile(generator_file_name):
        break

    if opt.algorithm == "orig_cond_wgan_gp":
        generators = []
        for i in range(opt.num_modalities):
            generators.append(cond_wgan_gp_gen(opt))
            generators[i].load_state_dict(
                torch.load(generator_file_name, map_location=torch.device(device))
            )
    else:
        generator = (
            mdmcgan_gen(opt) if opt.algorithm == "mdmcgan" else cond_wgan_gp_gen(opt)
        )
        generator.load_state_dict(
            torch.load(generator_file_name, map_location=torch.device(device))
        )

    gen_features, labels = generate_fake_features()
    f1_score = evaluate_fake_features(gen_features, labels)

    writer.writerow([i, f1_score])
    result_f.flush()

    i += opt.sample_interval

result_f.close()
