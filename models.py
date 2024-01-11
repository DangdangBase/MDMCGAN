import numpy as np
import torch
import torch.nn as nn


class Generator(nn.Module):
    # Generate conacatenated data [X_1, X_2, ...]
    def __init__(self, opt):
        super(Generator, self).__init__()
        self.opt = opt
        self.num_modalities = opt.num_modalities

        self.label_emb = nn.Embedding(opt.n_classes, opt.n_classes)
 
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.models = nn.ModuleList()
        for _ in range(self.num_modalities):
            model = nn.Sequential(
                *block(opt.latent_dim + opt.n_classes, 128, normalize=False),
                *block(128, 256),
                *block(256, 512),
                *block(512, 1024),
                nn.Linear(1024, int(np.prod(opt.img_shape))),
                nn.Tanh()
            )
            self.models.append(model)

    def forward(self, z, labels):
        generated_modalities = []
        for cur_model in self.models:
            gen_input = torch.cat((self.label_emb(labels), z), -1)
            img = cur_model(gen_input)
            img = img.view(img.shape[0], *self.opt.img_shape)
            generated_modalities.append(img)
        return generated_modalities


class Discriminator(nn.Module):
    def __init__(self, opt):
        super(Discriminator, self).__init__()

        self.label_embedding = nn.Embedding(opt.n_classes, opt.n_classes)

        # Copied from cgan.py
        self.model = nn.Sequential(
            nn.Linear(opt.n_classes + int(np.prod(opt.img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
        )

    def forward(self, img, labels, modal):
        # Concatenate label embedding and image to produce input
        d_in = torch.cat((img.view(img.size(0), -1), self.label_embedding(labels)), -1)
        validity = self.model(d_in)
        return validity
