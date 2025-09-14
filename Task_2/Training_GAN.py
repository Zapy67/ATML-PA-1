import torch
import torch.nn as nn
from torchvision.datasets import CIFAR10
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from architectures import VAE, GAN, vae_loss

device = "cuda" if torch.cuda.is_available() else "cpu"

# Constants
latent_dim = 128
img_channels = 3
feat_maps = 32
batch_size = 64
epochs = 3
lr = 2e-4
betas = (0.5, 0.999) 

# Setting up Dataset
transform = transforms.Compose([
    transforms.Resize(64),  # DCGAN expects 64x64
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])

print("=== Downloading CIFAR-10 ===")
trainset = CIFAR10(root="./data", train=True, download=True, transform=transform)
testset = CIFAR10(root="./data", train=False, download=True, transform=transform)

trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

# === GAN Training ===
model = GAN(latent_dim, img_channels, feat_maps, batch_size).to(device=device)

g_opt = optim.Adam(model.generator.parameters(), lr=lr, betas=betas)
d_opt = optim.Adam(model.discriminator.parameters(), lr=lr, betas=betas)
loss_fn = nn.BCELoss()

# Fixed noise, for monitoring progress
noise = torch.randn(batch_size, latent_dim, 1, 1, device=device)

# training loop
def train_GAN(model: GAN, g_opt: optim.Adam, d_opt: optim.Adam, loss_fn: nn.BCELoss, dataloader: DataLoader, epochs):
    model.generator.train()
    model.discriminator.train()

    all_g_losses = []
    all_d_losses = []

    for epoch in range(epochs):
        g_losses = []
        d_losses = []
        for (real_imgs, _) in dataloader:
            real_imgs = real_imgs.to(device)

            d_loss, g_loss = model.train_step(real_imgs, loss_fn=loss_fn, g_opt=g_opt, d_opt=d_opt)

            g_losses.append(g_loss)
            d_losses.append(d_loss)

        epoch_g_loss = sum(g_losses) / len(g_losses)
        epoch_d_loss = sum(d_losses) / len(d_losses)

        all_g_losses.append(epoch_g_loss)
        all_d_losses.append(epoch_d_loss)

        print(f"Epoch [{epoch+1}/{epochs}]  D_loss: {epoch_d_loss:.4f}  G_loss: {epoch_g_loss:.4f}")

    return all_g_losses, all_d_losses

print("=== Training GAN ===")
print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
print(f"Training on {device}")
print(f"Epochs: {epochs}")

g_losses, d_losses = train_GAN(model=model, g_opt=g_opt, d_opt=d_opt, loss_fn=loss_fn, dataloader=trainloader, epochs=epochs)

plt.plot(g_losses, label='Generator Loss')
plt.plot(d_losses, label='Discriminator Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('GAN Training Losses')

plt.savefig("GAN_Training_Losses.png", dpi=300, bbox_inches='tight')
plt.close()

# Save GAN Model
torch.save(model.state_dict(), "gan_weights.pth")