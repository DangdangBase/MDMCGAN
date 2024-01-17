import argparse
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt


from models.mdmcgan import Generator as mdmcgan_gen
from models.cond_wgan_gp import Generator as cond_wgan_gp_gens


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


if opt.algorithm == "orig_cond_wgan_gp":
    generators = []
    for i in range(opt.num_modalities):
        generators.append(cond_wgan_gp_gen(opt))
        generators[i].load_state_dict(
            torch.load(
                f"generator/{'non_iid' if opt.non_iid else 'iid'}_{opt.algorithm}_{i}"
            )
        )
else:
    generator = (
        mdmcgan_gen(opt) if opt.algorithm == "mdmcgan" else cond_wgan_gp_gen(opt)
    )
    generator.load_state_dict(
        torch.load(f"generator/{'non_iid' if opt.non_iid else 'iid'}_{opt.algorithm}")
    )

gen_label = 0

desired_ratios = [1 / 6 for _ in range(6)]
# desired_ratios = [1 if i == gen_label else 0 for i in range(6)]
total_samples = 6 * 10
counts = [int(ratio * total_samples) for ratio in desired_ratios]


z = Tensor(np.random.normal(0, 1, (total_samples, opt.latent_dim)))
labels = torch.cat([torch.full((count,), i) for i, count in enumerate(counts)])

with torch.no_grad():
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


processed_folder = "./UCI_HAR/processed_data"

y = np.load(f"{processed_folder}/labels.npy")

x_acc = np.load(f"{processed_folder}/acc_features.npy")
x_acc = np.squeeze(x_acc)

x_gyro = np.load(f"{processed_folder}/gyro_features.npy")
x_gyro = np.squeeze(x_gyro)

x = np.concatenate([x_acc, x_gyro], axis=2)


X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.1, random_state=0)


gen_features = gen_features.numpy()
gen_features = np.squeeze(gen_features)


for i in range(6):
    gen_label_idx = np.where(labels == i)[0]
    cur_gen_features = gen_features[gen_label_idx]

    real_label_idx = np.where(Y_train == i)[0]
    selected_idx = np.random.choice(
        real_label_idx, cur_gen_features.shape[0], replace=False
    )
    modifying_items = X_train[selected_idx]

    X_train = np.delete(X_train, selected_idx, axis=0)
    Y_train = np.delete(Y_train, selected_idx, axis=0)

    gen_split = np.split(cur_gen_features, 2, axis=2)
    real_split = np.split(modifying_items, 2, axis=2)

    new_x = np.concatenate(
        [
            np.concatenate([gen_split[0], real_split[1]], axis=2),
            np.concatenate([gen_split[1], real_split[0]], axis=2),
        ],
        axis=0,
    )

    X_train = np.concatenate([X_train, new_x], axis=0)
    Y_train = np.concatenate([Y_train, [i for _ in range(len(new_x))]], axis=0)


shuffler = np.random.permutation(len(X_train))
X_train = X_train[shuffler]
Y_train = Y_train[shuffler]

X_train = np.concatenate(np.moveaxis(X_train, 2, 0), axis=1)
X_test = np.concatenate(np.moveaxis(X_test, 2, 0), axis=1)


params = {
    "max_depth": [6, 8, 10],
    "n_estimators": [50, 100, 200],
    "min_samples_leaf": [8, 12],
    "min_samples_split": [8, 12],
}

rf_clf = RandomForestClassifier(random_state=0, n_jobs=-1)
grid_cv = GridSearchCV(
    rf_clf, param_grid=params, cv=2, n_jobs=-1, scoring="f1_weighted"
)
grid_cv.fit(X_train, Y_train)

print(grid_cv.best_params_)
print(grid_cv.score(X_test, Y_test))
