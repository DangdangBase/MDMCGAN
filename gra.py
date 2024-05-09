import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt


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

parser.add_argument("--avg_interval", type=int, default=4, help="interval of graph smoothing")
opt = parser.parse_args()

opt.feature_shape = (opt.channels, opt.feature_size, opt.feature_num)
opt.n_classes = 6

print(opt)

cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda" if cuda else "cpu")

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


def interval_avg(data:np.ndarray, interval=4):
    # 8개 datapoint로 평균
    if data.ndim == 3:
        data = data.reshape((data.shape[0], data.shape[1]))
    new_points = data.shape[1] // interval
    if data.shape[1] % interval > 0:
        new_points += 1
    new_data=np.zeros((data.shape[0], new_points))
    for i in range(new_points):
        if i < new_points-1:
            new_data[:,i] = data[:,i*interval:(i+1)*interval].mean(axis=1)
        else:
            new_data[:,i] = data[:,i*interval:].mean(axis=1)

    return new_data



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

desired_ratios = [1 if i == gen_label else 0 for i in range(6)]
total_samples = 6
counts = [int(ratio * total_samples) for ratio in desired_ratios]

"""
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
        gen_features = gen_features.squeeze()

        tmp = np.mean(gen_features.numpy(), axis=2)
        tmp = interval_avg(tmp, opt.avg_interval)

        for i in range(6):
            plt.subplot(6, 2, i + 1)
            plt.plot(tmp[i])
            plt.title(f"real_{i}")

    elif opt.algorithm == "cond_wgan_gp":
        gen_features = generator(z, labels)
        gen_copy = gen_features.clone().detach()
        gen_features = torch.cat([gen_features, gen_copy], dim=3)
    else:
        gen_features_list = []
        for i in range(opt.num_modalities):
            gen_features_list.append(generators[i](z, labels))
        gen_features = torch.cat(gen_features_list, dim=3)
"""

processed_folder = "UCI_HAR/processed_data"

labels = np.load(f"{processed_folder}/labels.npy")
acc_features = np.load(f"{processed_folder}/acc_features.npy")
gyro_features = np.load(f"{processed_folder}/gyro_features.npy")

gen_label_idx = np.where(labels == gen_label)[0]

labels_filtered = labels[gen_label_idx]
acc_features = acc_features[gen_label_idx]
gyro_features = gyro_features[gen_label_idx]


data = np.concatenate([acc_features, gyro_features], axis=3)
data = np.squeeze(data)
data = np.mean(data, axis=0)
tmp = np.hsplit(data, 6)
tmp = np.array(tmp)
tmp = interval_avg(tmp, opt.avg_interval)

plt.figure(figsize=(16,4))
for i in range(3):
    plt.subplot(1, 3, i + 1)
    plt.xticks([], [])
    plt.yticks([], [])
    if i == 1:
        plt.plot(tmp[i + 3], 'r')
    else:
        plt.plot(tmp[i + 3])
plt.show()
