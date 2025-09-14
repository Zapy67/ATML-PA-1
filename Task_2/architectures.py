import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, latent_dim=128):
        super(VAE, self).__init__()
        # Encoder
        self.enc_conv1 = nn.Conv2d(3, 32, 4, stride=2, padding=1)  # 32x32 -> 16x16
        self.enc_conv2 = nn.Conv2d(32, 64, 4, stride=2, padding=1) # 16x16 -> 8x8
        self.enc_fc = nn.Linear(64*8*8, 256)
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

        # Decoder
        self.dec_fc = nn.Linear(latent_dim, 256)
        self.dec_fc2 = nn.Linear(256, 64*8*8)
        self.dec_deconv1 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1) # 8x8 -> 16x16
        self.dec_deconv2 = nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1)  # 16x16 -> 32x32

    def encode(self, x):
        x = F.relu(self.enc_conv1(x))
        x = F.relu(self.enc_conv2(x))
        x = x.view(x.size(0), -1)
        h = F.relu(self.enc_fc(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h = F.relu(self.dec_fc(z))
        h = F.relu(self.dec_fc2(h))
        h = h.view(-1, 64, 8, 8)
        h = F.relu(self.dec_deconv1(h))
        x_recon = torch.sigmoid(self.dec_deconv2(h))
        return x_recon

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def vae_loss(x_reconstruction, x, mu, logvar, epoch=None, epochs=None, beta=1.0):
    # recon_loss = F.mse_loss(x_reconstruction, x, reduction='sum')
    recon_loss = F.binary_cross_entropy(x_reconstruction, x, reduction="sum")
    kl_loss = 0.5 * torch.sum(mu.pow(2) + logvar.exp() - logvar - 1)

    # KL Annealing
    if epoch is not None and epochs is not None:
        kl_weight = beta * epoch/epochs
    else:
        kl_weight = 1.0

    total_loss = (recon_loss + kl_weight * kl_loss) / x.size(0)
    recon_loss = recon_loss / x.size(0)
    kl_loss = kl_loss / x.size(0)

    return total_loss, recon_loss, kl_loss
    
class GAN(nn.Module):
    def __init__(self, latent_dim=128, img_channels=3, feat_maps=32, batch_size=64):
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        
        # Generator
        self.generator = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, feat_maps*8, 4, 1, 0, bias=True),
            nn.BatchNorm2d(feat_maps*8),
            nn.ReLU(True),

            nn.ConvTranspose2d(feat_maps*8, feat_maps*4, 4, 2, 1, bias=True),
            nn.BatchNorm2d(feat_maps*4),
            nn.ReLU(True),

            nn.ConvTranspose2d(feat_maps*4, feat_maps*2, 4, 2, 1, bias=True),
            nn.BatchNorm2d(feat_maps*2),
            nn.ReLU(True),

            nn.ConvTranspose2d(feat_maps*2, feat_maps, 4, 2, 1, bias=True),
            nn.BatchNorm2d(feat_maps),
            nn.ReLU(True),

            nn.ConvTranspose2d(feat_maps, img_channels, 4, 2, 1, bias=True),
            nn.Tanh()
        )

        # Discriminator
        self.discriminator = nn.Sequential(
            nn.Conv2d(img_channels, feat_maps, 4, 2, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feat_maps, feat_maps*2, 4, 2, 1, bias=True),
            nn.BatchNorm2d(feat_maps*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feat_maps*2, feat_maps*4, 4, 2, 1, bias=True),
            nn.BatchNorm2d(feat_maps*4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feat_maps*4, feat_maps*8, 4, 2, 1, bias=True),
            nn.BatchNorm2d(feat_maps*8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feat_maps*8, 1, 4, 1, 0, bias=True),
            nn.Sigmoid()
        )
    
    def generate(self, z):
        return self.generator(z)

    def discriminate(self, x):
        return self.discriminator(x).view(-1, 1).squeeze(1)
        
    def forward(self, x):
        z = torch.randn(self.batch_size, self.latent_dim, 1, 1, device=x.device)

        x_hat = self.generate(z)
        
        return self.discriminate(x), self.discriminate(x_hat.detach())  # detach so G isn't updated here

    def train_step(self, x, loss_fn, g_opt, d_opt):
        batch_size = x.size(0)
        # === Train Discriminator ===
        self.discriminator.zero_grad()

        # Real labels = 1, Fake labels = 0
        real_labels = torch.ones(batch_size, device=x.device)
        fake_labels = torch.zeros(batch_size, device=x.device)

        # discriminator result
        out_real, out_fake = self.forward(x)

        d_real_loss = loss_fn(out_real, real_labels)
        d_fake_loss = loss_fn(out_fake, fake_labels)

        # Combine & update D
        loss_d = d_real_loss + d_fake_loss
        loss_d.backward()
        d_opt.step()

        # === Train Generator ===
        self.generator.zero_grad()

        # Want D to classify fakes as real
        z = torch.randn(batch_size, self.latent_dim, 1, 1, device=x.device)
        fake_imgs = self.generate(z)
        out_fake_g = self.discriminate(fake_imgs)
        loss_g = loss_fn(out_fake_g, real_labels)

        loss_g.backward()
        g_opt.step()

        return loss_d.item(), loss_g.item()
