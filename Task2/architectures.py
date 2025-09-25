import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, latent_dim=128):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        # Encoder
        self.enc_conv1 = nn.Conv2d(3, 32, 3, stride=2, padding=1) # 32x32 -> 16x16
        self.enc_conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1) # 16x16 -> 8x8
        self.enc_conv3 = nn.Conv2d(64, 128, 3, stride=2, padding=1) # 8x8 -> 4x4
        self.enc_fc = nn.Linear(128*4*4, 512)
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)

        # Decoder
        self.dec_fc1 = nn.Linear(latent_dim, 128*4*4) # [batch, 2048]
        self.dec_deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 8x8 <- 4x4 [batch, 64, 8, 8]
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),   # 16x16 <- 8x8 [batch, 32, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),    # 32x32 <- 16x16 [batch, 3, 32, 32]
        )

    def encode(self, x):
        # x [batch, 3, 32, 32]
        x = F.relu(self.enc_conv1(x))
        x = F.relu(self.enc_conv2(x))
        x = F.relu(self.enc_conv3(x))
        x = x.view(x.size(0), -1)
        h = F.relu(self.enc_fc(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar # [batch, latent_dim]

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        # z [batch, latent_dim]
        h = F.relu(self.dec_fc1(z))
        h = h.view(-1, 128, 4, 4)
        x_recon = F.relu(self.dec_deconv(h))
        return x_recon # [batch, 3, 32, 32]

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def vae_loss(x_reconstruction, x, mu, logvar, epoch=None, epochs=None, kl_annealing=True, beta=0.0075):
    recon_loss = F.mse_loss(x_reconstruction, x, reduction="sum") / x.size(0)
    kl_loss = 0.5 * torch.sum(mu.pow(2) + logvar.exp() - logvar - 1) / x.size(0)

    # KL Annealing
    if kl_annealing:
        kl_weight = beta * min(1.0, 2*epoch/epochs)
    else:
        kl_weight = 1.0

    total_loss = (recon_loss + kl_weight * kl_loss)
    recon_loss = recon_loss
    kl_loss = kl_loss

    return total_loss, recon_loss, kl_loss
    
class GAN(nn.Module):
    def __init__(self, latent_dim=128, img_channels=3, feat_maps=32, batch_size=64, basic=True):
        super(GAN, self).__init__()
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

            nn.Conv2d(feat_maps, img_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Tanh()
        )

        # Discriminator
        
        self.disc_features = nn.Sequential(
            nn.Conv2d(img_channels, feat_maps, 4, 2, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feat_maps, feat_maps*2, 4, 2, 1, bias=True),
            nn.BatchNorm2d(feat_maps*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feat_maps*2, feat_maps*4, 4, 2, 1, bias=True),
            nn.BatchNorm2d(feat_maps*4),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Head (final classifier from features → prob)
        self.disc_head = nn.Sequential(
            nn.Conv2d(feat_maps*4, 1, 4, 1, 0, bias=True),
            nn.Sigmoid()
        )
    
    def generate(self, z):
        return self.generator(z)

    def discriminate(self, x, return_feats=False):
        feats = self.disc_features(x)                     # extract feature maps
        out = self.disc_head(feats).view(-1, 1).squeeze(1)  # final prob
        if return_feats:
            return out, feats
        return out

        
    def forward(self, z):
        return self.generate(z)

    def train_step(self, x, loss_fn, g_opt, d_opt,
                   epoch=None, sigma_start=0.05, sigma_end=0.01, anneal_epochs=30, basic=False):
        batch_size = x.size(0)

        # === Noise schedule (annealed) ===s

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
        if not basic and sigma > 0:
            x = x + sigma * torch.randn_like(x)
            x_hat = x_hat.detach() + sigma * torch.randn_like(x_hat)

            # clamp to valid range for tanh-scaled images
            x = x.clamp(-1.0, 1.0)
            x_hat = x_hat.clamp(-1.0, 1.0)
        else:
            x_hat = x_hat.detach()

        out_real, out_fake = self.discriminate(x), self.discriminate(x_hat) # detach so G isn't updated here

        if basic:
            real_labels = torch.ones_like(out_real, device=x.device)
            fake_labels = torch.zeros_like(out_fake, device=x.device)
        else:
            # Label smoothing: real=0.9
            real_labels = torch.full((batch_size,), 0.9, device=x.device)
            fake_labels = torch.zeros_like(out_fake, device=x.device)

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
        
        out_fake_g = self.discriminate(fake_imgs)
        target_labels = torch.ones_like(out_fake_g, device=x.device)
        loss_g = loss_fn(out_fake_g, target_labels)
        
        if not basic:
            # --- Feature Matching ---
            _, real_feats = self.discriminate(x, return_feats=True)
            _, fake_feats = self.discriminate(fake_imgs, return_feats=True)

            # match mean feature activations
            loss_g += 0.3 * torch.mean((real_feats.mean(dim=0) - fake_feats.mean(dim=0)) ** 2)

        loss_g.backward()
        g_opt.step()

        return loss_d.item(), loss_g.item()
