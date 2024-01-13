import numpy as np
import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()
        self.opt = opt
        self.num_modalities = opt.num_modalities

        self.label_emb = nn.Embedding(opt.n_classes, opt.n_classes)
        self.modal_emb = nn.Embedding(opt.num_modalities, opt.num_modalities)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(
                opt.latent_dim + opt.n_classes + opt.num_modalities,
                128,
                normalize=False,
            ),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(opt.feature_shape))),
            nn.Tanh()
        )

    def forward(self, z, labels, modal):
        emb = torch.cat((self.label_emb(labels), self.modal_emb(modal)), -1)
        gen_input = torch.cat((emb, z), -1)

        gen_feature = self.model(gen_input)
        gen_feature = gen_feature.view(gen_feature.shape[0], *self.opt.feature_shape)
        return gen_feature


class Discriminator(nn.Module):
    def __init__(self, opt):
        super(Discriminator, self).__init__()

        self.label_emb = nn.Embedding(opt.n_classes, opt.n_classes)
        self.modal_emb = nn.Embedding(opt.num_modalities, opt.num_modalities)

        self.model = nn.Sequential(
            nn.Linear(
                opt.n_classes + opt.num_modalities + int(np.prod(opt.feature_shape)),
                512,
            ),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
        )

    def forward(self, feature, labels, modal):
        emb = torch.cat((self.label_emb(labels), self.modal_emb(modal)), -1)
        d_in = torch.cat((feature.view(feature.size(0), -1), emb), -1)

        validity = self.model(d_in)
        return validity
