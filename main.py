import argparse
import os
import numpy as np
import math
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets

from models import Generator, Discriminator

os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--n_epochs", type=int, default=150, help="number of epochs of training"
)
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
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
    "--img_size", type=int, default=28, help="size of each image dimension"
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
    default="mnist_c",
    help="dataset to use",
)
parser.add_argument(
    "--num_modalities",
    type=int,
    default=5,
    help="number of modalities",
)
opt = parser.parse_args()
print(opt)

opt.img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False
opt.n_classes = 10

# Loss weight for gradient penalty
lambda_gp = 10

# Initialize generator and discriminator
generator = Generator(opt)
discriminator_1 = Discriminator(opt)
discriminator_2 = Discriminator(opt)
discriminator_3 = Discriminator(opt)
discriminator_4 = Discriminator(opt)
discriminator_5 = Discriminator(opt)
discriminators = [
    discriminator_1,
    discriminator_2,
    discriminator_3,
    discriminator_4,
    discriminator_5,
]

if cuda:
    generator.cuda()
    discriminator_1.cuda()
    discriminator_2.cuda()
    discriminator_3.cuda()
    discriminator_4.cuda()
    discriminator_5.cuda()

# Configure data loader

if opt.dataset == "mnist_c":
    mnist_corrupted_folder = "./data/mnist_c"
    arr = ["brightness", "canny_edges", "glass_blur", "identity", "scale"]

    # Create a list to store the dataloaders
    dataloader = []

    transform = transforms.Compose([transforms.Normalize([0.5], [0.5])])

    for idx, modal in enumerate(arr, start=1):
        # Load data for the current corruption type
        imgs = torch.from_numpy(
            np.load(f"{mnist_corrupted_folder}/{modal}/train_images.npy")
        ).permute(0, 3, 1, 2)
        labels = torch.from_numpy(
            np.load(f"{mnist_corrupted_folder}/{modal}/train_labels.npy")
        )
        # Append numeric modality information to the dataset
        modal = torch.tensor([idx] * len(imgs))

        # Create a TensorDataset with numeric modality information
        imgs = transform(imgs.float())
        current_dataset = TensorDataset(imgs, labels, modal)

        # Create a DataLoader for the current corruption type
        current_dataloader = DataLoader(
            dataset=current_dataset, batch_size=opt.batch_size, shuffle=True
        )

        # Append the DataLoader to the list
        dataloader.append(current_dataloader)

    # Now, dataloaders is a list containing dataloaders for each corruption type, including numeric modality information

elif opt.dataset == "mnist":
    os.makedirs("data/mnist", exist_ok=True)
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "data/mnist",
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.Resize(opt.img_size),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ]
            ),
        ),
        batch_size=opt.batch_size,
        shuffle=True,
    )
elif opt.dataset == "fashion":
    os.makedirs("data/fashion-mnist", exist_ok=True)
    dataloader = torch.utils.data.DataLoader(
        datasets.FashionMNIST(
            "data/fashion-mnist",
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.Resize(opt.img_size),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ]
            ),
        ),
        batch_size=opt.batch_size,
        shuffle=True,
    )


# Optimizers
optimizer_G = torch.optim.Adam(
    generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2)
)
optimizer_D_1 = torch.optim.Adam(
    discriminator_1.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2)
)
optimizer_D_2 = torch.optim.Adam(
    discriminator_2.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2)
)
optimizer_D_3 = torch.optim.Adam(
    discriminator_3.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2)
)
optimizer_D_4 = torch.optim.Adam(
    discriminator_4.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2)
)
optimizer_D_5 = torch.optim.Adam(
    discriminator_5.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2)
)
optimizers = [optimizer_D_1, optimizer_D_2, optimizer_D_3, optimizer_D_4, optimizer_D_5]
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


def sample_image(n_row, batches_done, n):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    # Sample noise
    z = Tensor(np.random.normal(0, 1, (n_row**2, opt.latent_dim)))
    # Get labels ranging from 0 to n_classes for n rows
    labels = np.array([num for _ in range(n_row) for num in range(n_row)])
    with torch.no_grad():
        labels = LongTensor(labels)
        gen_imgs = generator(z, labels)
    save_image(
        gen_imgs[n].data, "images/%d.png" % batches_done, nrow=n_row, normalize=True
    )


def compute_gradient_penalty(D, real_samples, fake_samples, labels, modal):
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
    d_interpolates = D(interpolates, labels, modal)
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

batches_done = 0
for epoch in range(opt.n_epochs):
    for j in range(5):
        for i, (imgs, labels, modal) in enumerate(dataloader[j]):
            batch_size = imgs.shape[0]

            # Move to GPU if necessary
            real_imgs = imgs.type(Tensor)
            labels = labels.type(LongTensor)
            modal = modal.type(LongTensor)

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizers[j].zero_grad()

            # Sample noise and labels as generator input
            z = Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim)))

            # Generate a batch of images
            fake_imgs = generator(z, labels)

            # Real images
            real_validity = discriminators[j](real_imgs, labels, modal)

            fake_labels = torch.randint(0, 10, size=(labels.size(0),), device='cuda:0')
            fake_labels = torch.where(fake_labels == labels, torch.randint_like(fake_labels, 0, 10), fake_labels)
            fake_label = discriminators[j](real_imgs, fake_labels, modal)

            fake_modals = torch.randint(0, 5, size=(modal.size(0),), device='cuda:0')
            fake_modals = torch.where(fake_modals == modal, torch.randint_like(fake_modals, 0, 5), fake_modals)
            fake_modality = discriminators[j](real_imgs, labels, fake_modals)
            
            # Fake images
            fake_validity = discriminators[j](fake_imgs[j], labels, modal)

            # Gradient penalty
            gradient_penalty = compute_gradient_penalty(
                discriminators[j],
                real_imgs.data,
                fake_imgs[j].data,
                labels.data,
                modal.data,
            )
            # Adversarial loss
            d_loss = (
                -torch.mean(real_validity)
                + torch.mean(fake_validity)
                + lambda_gp * gradient_penalty
            )
            
            d_dis_loss = (
                torch.mean((real_validity - 1) ** 2)
                + torch.mean(fake_label ** 2)
                + torch.mean(fake_modality ** 2)
                + torch.mean(fake_validity ** 2)
            )

            d_loss.backward()
            optimizers[j].step()

            optimizer_G.zero_grad()

            # Train the generator every n_critic steps
            if i % opt.n_critic == 0:
                # -----------------
                #  Train Generator
                # -----------------

                # Generate a batch of images
                fake_imgs = generator(z, labels)
                # Loss measures generator's ability to fool the discriminator
                # Train on fake images
                fake_validity = discriminators[j](fake_imgs[j], labels, modal)
                g_loss = -torch.mean(fake_validity)

                g_loss.backward()
                optimizer_G.step()

                print(
                    "[Epoch %d/%d] [Batch %d/%d] [Order %d/%d] [D loss: %f] [G loss: %f]"
                    % (
                        epoch,
                        opt.n_epochs,
                        i,
                        len(dataloader[j]),
                        j,
                        5,
                        d_loss.item(),
                        g_loss.item(),
                    )
                )

                if batches_done % opt.sample_interval == 0:
                    sample_image(opt.n_classes, batches_done, j)
                    # save_image(fake_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)

                batches_done += opt.n_critic
