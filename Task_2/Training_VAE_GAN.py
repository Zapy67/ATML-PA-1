import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
import torch.optim as optim

from architectures import VAE, GAN

device = "cuda" if torch.cuda.is_available() else "cpu"

# Constants
latent_dim = 128
img_channels = 3
feat_maps = 32
batch_size = 64
epochs = 50
lr = 2e-4
betas = (0.5, 0.999) 

# Setting Up GAE Training
model = GAN(latent_dim, img_channels, feat_maps, batch_size)

g_opt = optim.Adam(model.generator.parameters(), lr=lr, betas=betas)
d_opt = optim.Adam(model.discriminator.parameters(), lr=lr, betas=betas)
loss_fn = nn.BCELoss()

# Fixed noise, for monitoring progress
noise = torch.randn(batch_size, latent_dim, 1, 1, device=device)

# training loop
def train_GAN(model: GAN, g_opt: optim.Adam, d_opt: optim.Adam, loss_fn: nn.BCELoss, dataloader: torch.utils.data.DataLoader, epochs):
    for epoch in epochs:
        for (real_imgs, _) in dataloader:
            real_imgs = real_imgs.to(device)
             # === Train Discriminator ===
            model.discriminator.zero_grad()

            # Real labels = 1, Fake labels = 0
            real_labels = torch.ones(real_imgs.size(0), device=device)
            fake_labels = torch.zeros(real_imgs.size(0), device=device)

            # Real images
            out_real = model.discriminate(real_imgs)
            loss_d_real = loss_fn(out_real, real_labels)

            # Fake images
            z = torch.randn(real_imgs.size(0), latent_dim, 1, 1, device=device)
            fake_imgs = model.generate(z)
            out_fake = model.discriminate(fake_imgs.detach())  # detach so G isn't updated here
            loss_d_fake = loss_fn(out_fake, fake_labels)

            # Combine & update D
            loss_d = loss_d_real + loss_d_fake
            loss_d.backward()
            d_opt.step()

            # === Train Generator ===
            model.generator.zero_grad()

            # Want D to classify fakes as real
            out_fake_g = model.discriminate(fake_imgs)
            loss_g = loss_fn(out_fake_g, real_labels)

            loss_g.backward()
            g_opt.step()