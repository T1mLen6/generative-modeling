import argparse
import os
from utils import get_args

import torch

from networks import Discriminator, Generator
import torch.nn.functional as F
from train import train_model


def compute_discriminator_loss(
    discrim_real, discrim_fake, discrim_interp, interp, lamb
):
    ##################################################################
    # TODO 1.3: Implement GAN loss for discriminator.
    # Do not use discrim_interp, interp, lamb. They are placeholders
    # for Q1.5.
    ##################################################################
    bce_loss = torch.nn.BCELoss()

    # Discriminator Loss (D Loss)
    # Loss on real data
    real_labels = torch.ones_like(discrim_real)  # Target labels for real data
    d_loss_real = bce_loss(torch.sigmoid(discrim_real), real_labels)

    # Loss on fake data
    fake_labels = torch.zeros_like(discrim_fake)  # Target labels for fake data
    d_loss_fake = bce_loss(torch.sigmoid(discrim_fake), fake_labels)

    # Total discriminator loss
    loss = d_loss_real + d_loss_fake
    ##################################################################
    #                          END OF YOUR CODE                      #
    ##################################################################
    return loss


def compute_generator_loss(discrim_fake):
    ##################################################################
    # TODO 1.3: Implement GAN loss for the generator.
    ##################################################################
    bce_loss = torch.nn.BCELoss()
    real_labels = torch.ones_like(discrim_fake)
    g_loss = bce_loss(torch.sigmoid(discrim_fake), real_labels)
    ##################################################################
    #                          END OF YOUR CODE                      #
    ##################################################################
    return loss


if __name__ == "__main__":
    args = get_args()
    gen = Generator().cuda()
    disc = Discriminator().cuda()
    prefix = "data_gan/"
    os.makedirs(prefix, exist_ok=True)

    train_model(
        gen,
        disc,
        num_iterations=int(3e4),
        batch_size=256,
        prefix=prefix,
        gen_loss_fn=compute_generator_loss,
        disc_loss_fn=compute_discriminator_loss,
        log_period=1000,
        amp_enabled=not args.disable_amp,
    )
