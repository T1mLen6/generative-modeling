import os

import torch
import torch.nn.functional as F
from utils import get_args

from networks import Discriminator, Generator
from train import train_model


def compute_discriminator_loss(
    discrim_real, discrim_fake, discrim_interp, interp, lamb
):
    ##################################################################
    # TODO: 1.4: Implement LSGAN loss for discriminator.
    # Do not use discrim_interp, interp, lamb. They are placeholders
    # for Q1.5.
    ##################################################################
    mse_loss = torch.nn.MSELoss()

 
    real_labels = torch.ones_like(discrim_real)  # Target labels for real data
    d_loss_real = mse_loss(discrim_real, real_labels)

    fake_labels = torch.zeros_like(discrim_fake)  # Target labels for fake data
    d_loss_fake = mse_loss(discrim_fake, fake_labels)

    loss = 0.5 * (d_loss_real + d_loss_fake)
    ##################################################################
    #                          END OF YOUR CODE                      #
    ##################################################################
    return loss


def compute_generator_loss(discrim_fake):
    ##################################################################
    # TODO: 1.4: Implement LSGAN loss for generator.
    ##################################################################
    mse_loss = torch.nn.MSELoss()


    real_labels = torch.ones_like(discrim_fake)  # Target labels for real data
    loss = mse_loss(discrim_fake, real_labels)
    ##################################################################
    #                          END OF YOUR CODE                      #
    ##################################################################
    return loss

if __name__ == "__main__":
    args = get_args()
    gen = Generator().cuda()
    disc = Discriminator().cuda()
    prefix = "data_ls_gan/"
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
