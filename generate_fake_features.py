import argparse
import numpy as np
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
    choices=["mdmcgan", "cond_wgan_gp"],
    default="mdmcgan",
    help="feature generating algorithm to use",
)
parser.set_defaults(non_iid=True)
opt = parser.parse_args()

opt.feature_num = 3 if opt.algorithm == "mdmcgan" else 6
opt.feature_shape = (opt.channels, opt.feature_size, opt.feature_num)
opt.n_classes = 6

print(opt)

cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda" if cuda else "cpu")

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


generator = mdmcgan_gen(opt) if opt.algorithm == "mdmcgan" else cond_wgan_gp_gen(opt)
generator.load_state_dict(
    torch.load(f"generator/{'non_iid' if opt.non_iid else 'iid'}_{opt.algorithm}")
)


desired_ratios = [0.1, 0.2, 0.3, 0.2, 0.1, 0.1]
total_samples = 500
counts = [int(ratio * total_samples) for ratio in desired_ratios]


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
    else:
        gen_features = generator(z, labels)

np.save(
    f"UCI_HAR/gen_data/{'non_iid' if opt.non_iid else 'iid'}_{opt.algorithm}",
    gen_features.numpy(),
)
np.save("UCI_HAR/gen_data/labels", labels.numpy())
