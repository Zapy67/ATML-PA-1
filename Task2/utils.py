import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

from architectures import GAN, VAE, vae_loss

device = "cuda" if torch.cuda.is_available() else "cpu"

class LinearInterpolator:
    """
    Example Usage:
    (All Dims)
    vae_interpolator = LinearInterpolator(vae_model, device)
    z1 = torch.randn(vae_model.latent_dim)
    z2 = torch.randn(vae_model.latent_dim)
    video = vae_interpolator.inter(z1, z2)

    (Specific Dims (example = first dim))
    gan_interpolator = LinearInterpolator(gan_model, device)
    z1 = torch.randn(gan_model.latent_dim)
    z2 = torch.randn(gan_model.latent_dim)
    video = gan_interpolator.inter(z1, z2, vary_dims=[0])

    Make sure vary_dims is passed valid dimensions (0 <= dim < latent_dim)
    """
    def __init__(self, model: VAE | GAN, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.model.eval()

    def inter(self, z1, z2, vary_dims=None):
        z1 = z1.clone().to(self.device)
        z2 = z2.clone().to(self.device)

        if vary_dims is None:
            vary_dims = list(range(z1.shape[0]))

        latent_codes = []
        
        for alpha in np.linspace(0, 1, num=128):
            z = z1.clone()
            z[vary_dims] = z1[vary_dims] * (1-alpha) + z2[vary_dims] * alpha
            latent_codes.append(z)
        latent_codes = torch.stack(latent_codes).to(self.device)

        with torch.inference_mode():
            if isinstance(self.model, VAE):
                decoded = self.model.decode(latent_codes).reshape(-1, 3, 32, 32).cpu()
            else:
                latent_codes = latent_codes.view(latent_codes.size(0), latent_codes.size(1), 1, 1)
                decoded = self.model.generate(latent_codes).reshape(-1, 3, 32, 32).cpu()

        decoded = decoded.permute(0, 2, 3, 1)
        
        fig, ax = plt.subplots()
        ims = [[ax.imshow(im, animated=True)] for im in decoded.numpy()]
        ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,repeat_delay=1000)
        plt.close(fig)
        from matplotlib import rc
        rc('animation', html='jshtml')
        return ani

import torchvision.models as models
import torch.optim as optim

class GAN_Inversion:
    """
    GAN_Inversion performs inversion of real images into a GAN’s
    latent space. 

    GANs map latent vectors (z) to images, but inversion solves
    the reverse problem: given a real image, find the latent code
    such that the GAN’s generator can reconstruct that image.

    Why it matters:
        - Enables editing of real images via latent manipulations
          (e.g., change color, add attributes).
        - Helps in image restoration tasks (inpainting,
          super-resolution).
        - Allows analysis of how well the GAN latent space captures
          real-world image distributions.

    Methods:
        - Optimization-based inversion: iteratively adjusts latent
          vectors to minimize the difference between the generated
          and target image.
        - Encoder-based inversion: trains an auxiliary encoder to
          directly predict latent vectors from images.
        - Hybrid approaches: combine encoder initialization with
          optimization refinement.

    In short, GAN inversion bridges the gap between real images
    and the GAN’s imagination space, enabling reconstruction and
    controllable editing.
    """
    def __init__(self, model: GAN):
        self.model = model
        self.device = device if device is not None else next(model.parameters()).device

    def _init_vgg(self):
        """Simple VGG feature extractor for perceptual loss (uses conv features)."""
        vgg = models.vgg16(pretrained=True).features.eval().to(self.device)
        for p in vgg.parameters():
            p.requires_grad = False
        # We'll use a few layers (e.g., relu1_2, relu2_2, relu3_3)
        # Indices correspond to torchvision's VGG implementation
        layer_idx = {'relu1_2': 3, 'relu2_2': 8, 'relu3_3': 15}
        return vgg, layer_idx

    def _vgg_features(self, vgg, layer_idx, x):
        feats = {}
        cur = x
        for i, layer in enumerate(vgg):
            cur = layer(cur)
            for name, idx in layer_idx.items():
                if i == idx:
                    feats[name] = cur
        return feats

    def reconstruct(self, image, iters=1000, lr=0.05, use_perceptual=True, perc_weight=1.0,
                     pixel_weight=1.0, z_reg_weight=1e-4, latent_init='random', verbose=False):
        """
        Optimize latent z so that G(z) approximates `image`.

        Args:
            image: torch.Tensor, shape (C,H,W) or (1,C,H,W), expected in range [-1, 1].
            iters: number of optimization steps
            lr: learning rate for optimizer
            use_perceptual: whether to include VGG perceptual loss (requires torchvision)
            perc_weight: weight for perceptual loss
            pixel_weight: weight for pixel L2 loss
            z_reg_weight: weight for gaussian prior regularizer on z (||z||^2)
            latent_init: 'zeros' | 'random' | torch.Tensor(initial_z)
            verbose: print loss occasionally

        Returns:
            recon_img: tensor (1,C,H,W) in [-1,1]
            z_opt: optimized latent Tensor (1, latent_dim, 1, 1)
            logs: dict with final losses (pixel, perceptual, reg) if requested
        """

        self.model.eval()
        device = self.device

        # Ensure batch dimension
        if image.dim() == 3:
            image = image.unsqueeze(0)
        image = image.to(device)

        batch_size = image.size(0)
        latent_dim = self.model.latent_dim

        # Initialize latent
        if isinstance(latent_init, torch.Tensor):
            z = latent_init.to(device).detach().clone()
            if z.dim() == 2:
                z = z.view(batch_size, latent_dim, 1, 1)
        elif latent_init == 'random':
            z = torch.randn(batch_size, latent_dim, 1, 1, device=device)
        else:  # 'zeros' by default
            z = torch.zeros(batch_size, latent_dim, 1, 1, device=device)

        z = z.detach().clone().requires_grad_(True)

        optimizer = optim.Adam([z], lr=lr)

        # Optional perceptual network
        if use_perceptual:
            try:
                vgg, layer_idx = self._init_vgg()
            except Exception as e:
                # If VGG is unavailable, disable perceptual
                if verbose:
                    print("Warning: VGG initialization failed, disabling perceptual loss:", e)
                use_perceptual = False
                vgg = None
                layer_idx = None

        # Precompute real image vgg features if needed
        if use_perceptual:
            # VGG expects inputs in range [0,1] and normalized
            # Convert image from [-1,1] -> [0,1]
            image_vgg = (image + 1.0) / 2.0
            # Normalize with ImageNet stats
            mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1,3,1,1)
            std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1,3,1,1)
            image_vgg = (image_vgg - mean) / std
            real_feats = self._vgg_features(vgg, layer_idx, image_vgg)
        else:
            real_feats = None

        last_print = 0
        logs = {}

        for step in range(1, iters + 1):
            optimizer.zero_grad()

            # Generate image from current z
            gen = self.model.generate(z)      # expected range [-1,1]
            gen_clamped = gen.clamp(-1.0, 1.0)

            # Pixel L2 loss
            pixel_loss = F.mse_loss(gen_clamped, image) * pixel_weight

            # Perceptual loss (feature matching in VGG feature space)
            if use_perceptual:
                gen_vgg = (gen_clamped + 1.0) / 2.0
                gen_vgg = (gen_vgg - mean) / std
                gen_feats = self._vgg_features(vgg, layer_idx, gen_vgg)

                perc_loss = 0.0
                for k in real_feats:
                    # match mean activations per feature map
                    rf = real_feats[k].detach()
                    gf = gen_feats[k]
                    perc_loss = perc_loss + F.mse_loss(gf, rf)
                perc_loss = perc_weight * perc_loss
            else:
                perc_loss = torch.tensor(0.0, device=device)

            # Regularize z (encourage staying near prior)
            z_reg = z.pow(2).mean() * z_reg_weight

            loss = pixel_loss + perc_loss + z_reg
            loss.backward()
            optimizer.step()

            # Optional: small latent clipping can help with some GANs
            # z.data = torch.clamp(z.data, -3.0, 3.0)

            if verbose and (step % max(1, iters // 10) == 0 or step == 1):
                print(f"[{step}/{iters}] total={loss.item():.6f} pixel={pixel_loss.item():.6f} perc={perc_loss.item():.6f} zreg={z_reg.item():.6f}")
                last_print = step

        # Final outputs
        with torch.no_grad():
            recon = self.model.generate(z).clamp(-1.0, 1.0)

        logs['pixel_loss'] = pixel_loss.item()
        logs['perc_loss'] = perc_loss.item() if isinstance(perc_loss, torch.Tensor) else float(perc_loss)
        logs['z_reg'] = z_reg.item()

        # Return single-image outputs for convenience
        return recon.detach(), z.detach(), logs

class GAN_FID:
    """
    GAN_FID evaluates the quality of images generated by a GAN
    using the Frechet Inception Distance (FID).

    FID measures the similarity between the distribution of
    generated images and real images. It works by:
        1. Passing both real and generated images through a
           pretrained Inception-v3 network.
        2. Extracting activations (features) from a chosen layer.
        3. Modeling these features as multivariate Gaussians.
        4. Computing the Frechet distance between the two
           Gaussian distributions.

    A lower FID score indicates that the generated images are
    more similar to the real images in terms of distribution,
    thus reflecting better image quality and diversity.

    Usage:
        - Initialize the class with a dataset (e.g., CIFAR-10).
        - Generate a batch of images using a trained GAN.
        - Compute FID against real images from the dataset.

    This is a standard benchmark for evaluating GANs.
    """
    def __init__(self, generator_path, device, latent_dim=128, batch_size=64, num_fake=1000):
        self.device = device
        self.batch_size = batch_size
        self.num_fake = num_fake
        self.latent_dim = latent_dim

        # Data transforms for CIFAR10
        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
        ])

        cifar_test = datasets.CIFAR10(
            root="./data", train=False, transform=transform, download=True
        )
        self.test_loader = DataLoader(
            cifar_test, batch_size=batch_size, shuffle=False
        )

        self.GAN = GAN(latent_dim=latent_dim).to(self.device)
        self.GAN.load_state_dict(torch.load(generator_path, map_location=self.device))
        self.GAN.eval()

        self.fid = FrechetInceptionDistance(feature=2048).to(self.device)

    def compute_fid(self):
        self.fid.reset()

        for imgs, _ in self.test_loader:
            imgs = (imgs * 255).to(torch.uint8).to(self.device)
            self.fid.update(imgs, real=True)

        with torch.inference_mode():
            for _ in range(self.num_fake // self.batch_size):
                z = torch.randn(self.batch_size, self.latent_dim, 1, 1, device=self.device)
                fake_imgs = self.GAN.generate(z)

                # Rescale from [-1,1] → [0,255]
                fake_imgs = ((fake_imgs + 1) / 2 * 255).clamp(0, 255)
                fake_imgs = fake_imgs.to(torch.uint8)

                self.fid.update(fake_imgs, real=False)

        score = self.fid.compute().item()
        print(f"FID score for GAN: {score:.4f}")
        return score
