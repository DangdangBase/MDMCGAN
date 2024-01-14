import os
import numpy as np
import csv

import torch
import torch.autograd as autograd

from torch.utils.data import DataLoader, TensorDataset

from models.cond_wgan_gp import Generator, Discriminator
from utils import count_parameters
from arg_parser.cond_wgan_gp import opt

os.makedirs("generator", exist_ok=True)
os.makedirs("gen_features/cond_wgan_gp", exist_ok=True)


cuda = True if torch.cuda.is_available() else False


data_size = np.prod(opt.feature_shape)

# Loss weight for gradient penalty
lambda_gp = 10

# Initialize generator and discriminator
generator = Generator(opt)
discriminator = Discriminator(opt)

if cuda:
    generator.cuda()
    discriminator.cuda()


if opt.dataset == "uci_har":
    data_folder = f"UCI_HAR/{'filtered_data' if opt.non_iid else 'processed_data'}"
    arr = ["acc", "gyro"]

    labels = torch.from_numpy(np.load(f"{data_folder}/labels.npy"))

    features_list = []

    for modal in arr:
        features = torch.from_numpy(np.load(f"{data_folder}/{modal}_features.npy"))
        features_list.append(features)

    whole_features = torch.concat(features_list, dim=3)

    dataset = TensorDataset(whole_features, labels)
    dataloader = DataLoader(dataset=dataset, batch_size=opt.batch_size, shuffle=True)


# Optimizers
optimizer_G = torch.optim.Adam(
    generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2)
)
optimizer_D = torch.optim.Adam(
    discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2)
)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


def sample_feature(batches_done):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    # Sample noise
    z = Tensor(np.random.normal(0, 1, (opt.n_classes, opt.latent_dim)))
    # Get labels ranging from 0 to n_classes for n rows
    labels = np.array([f for f in range(opt.n_classes)])
    with torch.no_grad():
        labels = LongTensor(labels)
        gen_features = generator(z, labels)

    np.save("gen_features/cond_wgan_gp/%d" % batches_done, gen_features.numpy())


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

result_f = open("cond_wgan_gp_result.csv", "w")
writer = csv.writer(result_f)
writer.writerow(["Epoch", "Batch", "D loss", "G loss", "D workload", "G workload"])

d_workload = 0
g_workload = 0
batches_done = 0

for epoch in range(opt.n_epochs):
    for i, (features, labels) in enumerate(dataloader):
        batch_size = features.shape[0]

        # Move to GPU if necessary
        real_features = features.type(Tensor)
        labels = labels.type(LongTensor)

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Sample noise and labels as generator input
        z = Tensor(np.random.normal(0, 1, (features.shape[0], opt.latent_dim)))

        # Generate a batch of features
        fake_features = generator(z, labels)

        # Real features
        real_validity = discriminator(real_features, labels)
        # Fake features
        fake_validity = discriminator(fake_features, labels)
        # Gradient penalty
        gradient_penalty = compute_gradient_penalty(
            discriminator, real_features.data, fake_features.data, labels.data
        )
        # Adversarial loss
        d_loss = (
            -torch.mean(real_validity)
            + torch.mean(fake_validity)
            + lambda_gp * gradient_penalty
        )

        d_loss.backward()
        optimizer_D.step()

        optimizer_G.zero_grad()

        d_workload += 6 * opt.batch_size * count_parameters(discriminator)

        # Train the generator every n_critic steps
        if i % opt.n_critic == 0:
            # -----------------
            #  Train Generator
            # -----------------

            # Generate a batch of features
            fake_features = generator(z, labels)
            # Loss measures generator's ability to fool the discriminator
            # Train on fake features
            fake_validity = discriminator(fake_features, labels)
            g_loss = -torch.mean(fake_validity)

            g_loss.backward()
            optimizer_G.step()

            k = 1
            g_workload += opt.batch_size * (
                data_size * 1 + k * count_parameters(generator)
            )

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (
                    epoch,
                    opt.n_epochs,
                    i,
                    len(dataloader),
                    d_loss.item(),
                    g_loss.item(),
                )
            )

            if batches_done % opt.sample_interval == 0:
                sample_feature(batches_done)
                # save_feature(fake_imgs.data[:25], "features/%d.png" % batches_done, nrow=5, normalize=True)

                writer.writerow(
                    [epoch, i, d_loss.item(), g_loss.item(), d_workload, g_workload]
                )
                result_f.flush()

            batches_done += opt.n_critic

result_f.close()

# save model
torch.save(generator.state_dict(), "generator/cond_wgan_gp")
