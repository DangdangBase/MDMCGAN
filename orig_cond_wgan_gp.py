import os
import numpy as np
import csv

import torch
import torch.autograd as autograd

from torch.utils.data import DataLoader, TensorDataset

from models.cond_wgan_gp import Generator, Discriminator
from utils import count_parameters
from train_params import opt

os.makedirs("generator", exist_ok=True)
os.makedirs("gen_features/cond_wgan_gp", exist_ok=True)


cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda" if cuda else "cpu")


# Loss weight for gradient penalty
lambda_gp = 10

# Initialize generator and discriminator
generators = []
for _ in range(opt.num_modalities):
    G = Generator(opt)
    generators.append(G)

discriminators = []
for _ in range(opt.num_modalities):
    D = Discriminator(opt)
    discriminators.append(D)

if cuda:
    for G in generators:
        G.cuda()
    for D in discriminators:
        D.cuda()

# Configure data loader

if opt.dataset == "uci_har":
    uci_har_folder = f"UCI_HAR/{'filtered_data' if opt.non_iid else 'processed_data'}"
    arr = ["acc", "gyro"]

    labels = torch.from_numpy(np.load(f"{uci_har_folder}/labels.npy"))

    dataloader = []

    for modal in arr:
        features = torch.from_numpy(np.load(f"{uci_har_folder}/{modal}_features.npy"))

        current_dataset = TensorDataset(features, labels)
        dataset_size = len(current_dataset)

        data_size = torch.flatten(features[0]).size(0)

        train_dataloader = DataLoader(
            dataset=current_dataset, batch_size=opt.batch_size, shuffle=True
        )

        dataloader.append(train_dataloader)

else:
    raise NotImplementedError


optimizers_G = []
for G in generators:
    O = torch.optim.Adam(G.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizers_G.append(O)

optimizers_D = []
for D in discriminators:
    O = torch.optim.Adam(D.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizers_D.append(O)


Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


def compute_gradient_penalty(D, real_samples, fake_samples, labels):
    """Calculates the gradient penalty loss for WGAN GP.
    Warning: It doesn't compute the gradient w.r.t the labels, only w.r.t
    the interpolated real and fake samples, as in the WGAN GP paper.
    """
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    labels = LongTensor(labels)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(
        True
    )
    d_interpolates = D(interpolates, labels)
    fake = Tensor(real_samples.shape[0], 1).fill_(1.0)
    fake.requires_grad = False
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )
    gradients = gradients[0].view(gradients[0].size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


# ----------
#  Training
# ----------

result_f = open(
    f"{'non_iid' if opt.non_iid else 'iid'}_orig_cond_wgan_gp_result.csv", "w"
)
writer = csv.writer(result_f)
writer.writerow(["Epoch", "Batch", "D loss", "G loss", "D workload", "G workload"])

batches_done = 0
d_workload = 0
g_workload = 0
g_losses = [None for _ in range(opt.num_modalities)]

for epoch in range(opt.n_epochs):
    it_list = []
    for dl in dataloader:
        it_list.append(iter(dl))

    batch_num = len(it_list[0])

    for i in range(batch_num):
        is_critic = i % opt.n_critic == 0

        for j in range(opt.num_modalities):
            (features, labels) = next(it_list[j])
            batch_size = features.shape[0]

            real_features = features.type(Tensor)
            labels = labels.type(LongTensor)

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizers_D[j].zero_grad()

            # Sample noise and labels as generator input
            z = Tensor(np.random.normal(0, 1, (features.shape[0], opt.latent_dim)))

            # Generate a batch of features
            fake_features = generators[j](z, labels)

            # Real features
            real_validity = discriminators[j](real_features, labels)
            # Fake features
            fake_validity = discriminators[j](fake_features, labels)
            # Gradient penalty
            gradient_penalty = compute_gradient_penalty(
                discriminators[j], real_features.data, fake_features.data, labels.data
            )
            # Adversarial loss
            d_loss = (
                -torch.mean(real_validity)
                + torch.mean(fake_validity)
                + lambda_gp * gradient_penalty
            )

            d_loss.backward()
            optimizers_D[j].step()

            optimizers_G[j].zero_grad()

            if is_critic:
                # -----------------
                # Collect Generator loss
                # -----------------

                fake_features = generators[j](z, labels)
                fake_validity = discriminators[j](fake_features, labels)
                g_losses[j] = torch.mean((fake_validity - 1) ** 2)

        d_workload += 6 * opt.batch_size * count_parameters(discriminators[0])

        if is_critic:
            # -----------------
            #  Train Generator
            # -----------------

            for j in range(opt.num_modalities):
                optimizers_G[j].zero_grad()
                g_loss = g_losses[j]

                g_loss.backward()
                optimizers_G[j].step()

            # -----------------
            # Energy consumption
            # -----------------

            k = 1
            g_workload += opt.batch_size * (
                data_size * opt.num_modalities + k * count_parameters(generators[j])
            )

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (
                    epoch,
                    opt.n_epochs,
                    i,
                    batch_num,
                    d_loss.item(),
                    g_loss.item(),
                )
            )

            if batches_done % opt.sample_interval == 0:
                writer.writerow(
                    [epoch, i, d_loss.item(), g_loss.item(), d_workload, g_workload]
                )
                result_f.flush()

            batches_done += opt.n_critic

            for i in range(opt.num_modalities):
                torch.save(
                    generators[i].state_dict(),
                    f"generator/{'non_iid' if opt.non_iid else 'iid'}_orig_cond_wgan_gp_{i}",
                )


result_f.close()

# save model
for i in range(opt.num_modalities):
    torch.save(
        generators[i].state_dict(),
        f"generator/{'non_iid' if opt.non_iid else 'iid'}_orig_cond_wgan_gp_{i}",
    )
