import argparse
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier

parser = argparse.ArgumentParser()
parser.add_argument(
    "--algorithm",
    type=str,
    choices=["mdmcgan", "cond_wgan_gp"],
    default="mdmcgan",
    help="algorithm to evaluate",
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
parser.set_defaults(non_iid=True)
opt = parser.parse_args()
print(opt)

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


Y_gen = np.load(f"{generated_folder}/labels.npy")

X_gen = np.load(
    f"{generated_folder}/{'non_iid' if opt.non_iid else 'iid'}_{opt.algorithm}.npy"
)
X_gen = np.squeeze(X_gen)
X_gen = np.concatenate(np.moveaxis(X_gen, 2, 0), axis=1)


X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.1, random_state=0)

X_train = np.concatenate((X_train, X_gen), axis=0)
Y_train = np.concatenate((Y_train, Y_gen), axis=0)


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

# {'max_depth': 10, 'min_samples_leaf': 8, 'min_samples_split': 8, 'n_estimators': 200}
