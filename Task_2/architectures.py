import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, latent_dim=128):
        super(VAE, self).__init__()
        # Encoder
        self.enc_conv1 = nn.Conv2d(3, 32, 4, stride=2, padding=1) # 32x32 -> 16x16
        self.enc_conv2 = nn.Conv2d(32, 64, 4, stride=2, padding=1) # 16x16 -> 8x8
        self.enc_conv3 = nn.Conv2d(64, 128, 4, stride=2, padding=1) # 8x8 -> 4x4
        self.enc_fc = nn.Linear(128*4*4, 512)
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)

        # Decoder
        self.dec_fc1 = nn.Linear(latent_dim, latent_dim)
        self.dec_fc2 = nn.Linear(latent_dim, 512)
        self.dec_fc3 = nn.Linear(1024, 128*4*4)
        self.dec_deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 3, stride=2, padding=1, output_padding=1), # 8x8 <- 4x4
            nn.ConvTranspose2d(128, 128, 3, stride=1, padding=1), # 8->8
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=0), # 8->16
            nn.ConvTranspose2d(64, 64, 3, stride=1, padding=1), # 16->16
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=0), # 16->32
            nn.ConvTranspose2d(32, 32, 3, stride=1, padding=1), # 32->32
        )
        self.dec_deconv2 = nn.Conv2d(32, 3, 3, padding=1) # 32 -> 32 

    def encode(self, x):
        x = F.relu(self.enc_conv1(x))
        x = F.relu(self.enc_conv2(x))
        x = F.relu(self.enc_conv3(x))
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
        h = F.relu(self.dec_fc3(h))
        h = h.view(-1, 256, 4, 4)
        h = F.relu(self.dec_deconv(h))
        x_recon = torch.sigmoid(self.dec_deconv2(h))
        return x_recon

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def vae_loss(x_reconstruction, x, mu, logvar, epoch=None, epochs=None, beta=1.0):
    recon_loss = F.mse_loss(x_reconstruction, x, reduction="sum")
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
        super(GAN, self).__init__()
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        
        # Generator
        self.generator = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, feat_maps*4, 4, 1, 0, bias=True),
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

            nn.Conv2d(feat_maps*4, 1, 4, 1, 0, bias=True),
            nn.Sigmoid()
        )
    
    def generate(self, z):
        return self.generator(z)

    def discriminate(self, x):
        return self.discriminator(x).view(-1, 1).squeeze(1)
        
    def forward(self, z):
        return self.generate(z)

    def train_step(self, x, loss_fn, g_opt, d_opt,
                   epoch=None, sigma_start=0.05, sigma_end=0.0, anneal_epochs=20):
        batch_size = x.size(0)

        # === Noise schedule (annealed) ===
        if epoch is not None:
            t = min(epoch, anneal_epochs) / max(1, anneal_epochs)
            sigma = sigma_start * (1 - t) + sigma_end * t
        else:
            sigma = sigma_start  # fixed noise if no epoch passed

        # === Train Discriminator ===
        d_opt.zero_grad()

        # discriminator result
        z = torch.randn(batch_size, self.latent_dim, 1, 1, device=x.device)
        x_hat = self.forward(z)

        # Add lil bit of noise to real & fake before discrimination
        if sigma > 0:
            x = x + sigma * torch.randn_like(x)
            x_hat = x_hat.detach() + sigma * torch.randn_like(x_hat)

            # clamp to valid range for tanh-scaled images
            x = x.clamp(-1.0, 1.0)
            x_hat = x_hat.clamp(-1.0, 1.0)
        else:
            x_hat = x_hat.detach()

        out_real, out_fake = self.discriminate(x), self.discriminate(x_hat) # detach so G isn't updated here

        # Real labels = 1, Fake labels = 0
        real_labels = torch.full((batch_size,), 0.9, device=x.device)
        fake_labels = torch.zeros_like(out_fake)

        d_real_loss = loss_fn(out_real, real_labels)
        d_fake_loss = loss_fn(out_fake, fake_labels)

        # Combine & update D
        loss_d = d_real_loss + d_fake_loss
        loss_d.backward()
        d_opt.step()

        # === Train Generator ===
        g_opt.zero_grad()

        # Want D to classify fakes as real
        z = torch.randn(batch_size, self.latent_dim, 1, 1, device=x.device)
        fake_imgs = self.generate(z)
        out_fake_g = self.discriminate(fake_imgs)
        loss_g = loss_fn(out_fake_g, real_labels)

        loss_g.backward()
        g_opt.step()

        return loss_d.item(), loss_g.item()
