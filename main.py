import os
import numpy as np
import csv

import torch
import torch.autograd as autograd

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader, TensorDataset, Subset

from models import Generator, Discriminator
from arg_parser import opt

import pytorch_fid_wrapper as pfw

os.makedirs("images", exist_ok=True)


cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda" if cuda else "cpu")

pfw.set_config(batch_size=10, dims=192, device=device)

opt.img_shape = (opt.channels, opt.img_size, opt.img_size)
opt.n_classes = 10

# Loss weight for gradient penalty
lambda_gp = 10

# Initialize generator and discriminator
generator = Generator(opt)

discriminators = []
for _ in range(opt.num_modalities):
    D = Discriminator(opt)
    discriminators.append(D)

if cuda:
    generator.cuda()
    for D in discriminators:
        D.cuda()

# Configure data loader

if opt.dataset == "mnist_c":
    mnist_corrupted_folder = "./data/mnist_c"
    arr = ["canny_edges", "identity"]

    # Create a list to store the dataloaders
    dataloader = []
    validation_arrs = []

    transform = transforms.Compose([transforms.Normalize([0.5], [0.5])])

    for idx, modal in enumerate(arr):
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
        imgs = imgs / 255.0
        current_dataset = TensorDataset(imgs, labels, modal)
        dataset_size = len(current_dataset)
        train_size = int(dataset_size * 0.9)
        validation_size = int(dataset_size - train_size)

        train_dataset = Subset(current_dataset, range(train_size))

        data_size = torch.flatten(imgs[0]).size(0)

        # Create a DataLoader for the current corruption type
        train_dataloader = DataLoader(
            dataset=train_dataset, batch_size=opt.batch_size, shuffle=True
        )

        # Append the DataLoader to the list
        dataloader.append(train_dataloader)

        imgs_list = imgs[train_size:]
        imgs_list = imgs_list.expand(validation_size, 3, 28, 28)

        real_m, real_s = pfw.get_stats(imgs_list)
        validation_arrs.append((real_m, real_s))
else:
    raise NotImplementedError

# Optimizers
optimizer_G = torch.optim.Adam(
    generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2)
)

optimizers_D = []
for D in discriminators:
    O = torch.optim.Adam(D.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizers_D.append(O)


Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


def sample_image(n_row, batches_done):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    # Sample noise
    z = Tensor(np.random.normal(0, 1, (n_row * opt.num_modalities, opt.latent_dim))).to(
        device
    )
    # Get labels ranging from 0 to n_classes for n rows
    labels = torch.tensor(
        [num for _ in range(opt.num_modalities) for num in range(n_row)]
    ).to(device)
    modal = (
        torch.tensor([[idx] * n_row for idx in range(opt.num_modalities)])
        .flatten()
        .to(device)
    )

    with torch.no_grad():
        gen_imgs = generator(z, labels, modal)

    save_image(
        gen_imgs.data, "images/%d.png" % batches_done, nrow=n_row, normalize=True
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


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_evaluation():
    z = Tensor(
        np.random.normal(0, 1, (opt.n_classes * opt.num_modalities, opt.latent_dim))
    ).to(device)
    labels = torch.tensor(
        [num for _ in range(opt.num_modalities) for num in range(opt.n_classes)]
    ).to(device)
    modal = (
        torch.tensor([[idx] * opt.n_classes for idx in range(opt.num_modalities)])
        .flatten()
        .to(device)
    )

    with torch.no_grad():
        gen_imgs = generator(z, labels, modal)
        gen_imgs = gen_imgs.expand(opt.n_classes * opt.num_modalities, 3, 28, 28)

        imgs_list = torch.chunk(gen_imgs, opt.num_modalities)
        fids = []

        for i, (real_m, real_s) in enumerate(validation_arrs):
            fid = pfw.fid(imgs_list[i], real_m=real_m, real_s=real_s)
            fids.append(fid)

    return fids


# ----------
#  Training
# ----------

result_f = open("mdmcgan_result.csv", "w")
writer = csv.writer(result_f)
writer.writerow(
    [
        "Epoch",
        "Batch",
        "D loss",
        "G loss",
        "D workload",
        "G workload",
        "FID1",
        "FID2",
        "FID3",
        "FID4",
        "FID5",
    ]
)

batches_done = 0
d_workload = 0
g_workload = 0
g_losses = [None for _ in range(opt.num_modalities)]

for epoch in range(opt.n_epochs):
    it_list = []
    for dl in dataloader:
        it_list.append(iter(dl))

    data_len = len(dataloader[0])

    for i in range(data_len):
        is_critic = i % opt.n_critic == 0

        for j in range(opt.num_modalities):
            (imgs, labels, modal) = next(it_list[j])
            batch_size = imgs.shape[0]

            # Move to GPU if necessary
            real_imgs = imgs.type(Tensor)
            labels = labels.type(LongTensor)
            modal = modal.type(LongTensor)

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizers_D[j].zero_grad()

            # Sample noise and labels as generator input
            z = Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim)))

            # Generate a batch of images
            fake_imgs = generator(z, labels, modal)

            # Real images
            real_validity = discriminators[j](real_imgs, labels, modal)

            fake_labels = torch.randint(0, opt.n_classes, size=(labels.size(0),), device=device)
            fake_labels = torch.where(
                fake_labels == labels,
                torch.randint_like(fake_labels, 0, opt.n_classes),
                fake_labels,
            )
            fake_label = discriminators[j](real_imgs, fake_labels, modal)

            fake_modals = torch.randint(0, opt.num_modalities, size=(modal.size(0),), device=device)
            fake_modals = torch.where(
                fake_modals == modal, torch.randint_like(fake_modals, 0, opt.num_modalities), fake_modals
            )
            fake_modality = discriminators[j](real_imgs, labels, fake_modals)

            # Fake images
            fake_validity = discriminators[j](fake_imgs, labels, modal)
            # Gradient penalty
            gradient_penalty = compute_gradient_penalty(
                discriminators[j],
                real_imgs.data,
                fake_imgs.data,
                labels.data,
                modal.data,
            )
            # Adversarial loss

            d_dis_loss = (
                torch.mean((real_validity - 1) ** 2)
                + torch.mean(fake_label**2)
                + torch.mean(fake_modality**2)
                + torch.mean(fake_validity**2)
                + lambda_gp * gradient_penalty
            )

            d_dis_loss.backward()
            optimizers_D[j].step()

            if is_critic:
                # -----------------
                # Collect Generator loss
                # -----------------

                # Generate a batch of images
                fake_imgs = generator(z, labels, modal)
                # Loss measures generator's ability to fool the discriminator
                # Train on fake images
                fake_validity = discriminators[j](fake_imgs, labels, modal)
                g_losses[j] = -torch.mean(fake_validity)

        d_workload += 6 * opt.batch_size * count_parameters(discriminators[0])

        if is_critic:
            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            exp_losses = [torch.exp(loss) for loss in g_losses]
            exp_loss_sum = sum(exp_losses)

            g_loss = 0
            for k in range(opt.num_modalities):
                g_loss += exp_losses[k] * g_losses[k] / exp_loss_sum

            g_loss.backward()
            optimizer_G.step()

            # -----------------
            # Energy consumption
            # -----------------

            k = 1
            g_workload += opt.batch_size * (
                data_size * opt.num_modalities + k * count_parameters(generator)
            )

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (
                    epoch,
                    opt.n_epochs,
                    i,
                    data_len,
                    d_dis_loss.item(),
                    g_loss.item(),
                )
            )

            if batches_done % opt.sample_interval == 0:
                sample_image(opt.n_classes, batches_done)
                # save_image(fake_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)

                fids = model_evaluation()
                print("FIDs:", fids)

                writer.writerow(
                    [epoch, i, d_dis_loss.item(), g_loss.item(), d_workload, g_workload]
                    + fids
                )
                result_f.flush()

            batches_done += opt.n_critic

result_f.close()
