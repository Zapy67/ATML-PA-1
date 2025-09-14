import torch
import torch.nn as nn
from torchvision.datasets import CIFAR10
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from architectures import VAE, vae_loss

device = "cuda" if torch.cuda.is_available() else "cpu"

# Constants
batch_size = 64
epochs = 15
lr = 3e-4
betas = (0.9, 0.999)

# Setting up Dataset
transform = transforms.Compose([
    transforms.ToTensor()
])

print("=== Downloading CIFAR-10 ===")
trainset = CIFAR10(root="./data", train=True, download=True, transform=transform)
testset = CIFAR10(root="./data", train=False, download=True, transform=transform)

trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

# === Training VAE ===
def train_vae_step(model: VAE, train_loader, optimizer: optim.Adam, epoch, epochs, beta=1.0):
    model.train()
    train_loss = 0
    train_recon_loss = 0
    train_kl_loss = 0

    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()

        # Forward pass
        recon_data, mu, logvar = model(data)

        # Compute loss
        loss, recon_loss, kl_loss = vae_loss(recon_data, data, mu, logvar, epoch, epochs, beta)
        # loss = vae_loss(recon_data, data, mu, logvar, epoch, epochs, beta)

        # Backward pass
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_recon_loss += recon_loss.item()
        train_kl_loss += kl_loss.item()

        # if batch_idx % 100 == 0:
        #     print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
        #           f'({100. * batch_idx / len(train_loader):.0f}%)]\t'
        #           f'Loss: {loss.item() / len(data):.6f} '
        #           f'(Recon: {recon_loss.item() / len(data):.6f}, '
        #           f'KL: {kl_loss.item() / len(data):.6f})')

    avg_loss = train_loss / len(train_loader.dataset)
    avg_recon_loss = train_recon_loss / len(train_loader.dataset)
    avg_kl_loss = train_kl_loss / len(train_loader.dataset)

    print(f'====> Epoch: {epoch} Average loss: {avg_loss:.4f} '
          f'(Recon: {avg_recon_loss:.4f}, KL: {avg_kl_loss:.4f})')

    return avg_loss, avg_recon_loss, avg_kl_loss

def test_vae_step(model: VAE, test_loader, beta=1.0):
    model.eval()
    test_loss = 0
    test_recon_loss = 0
    test_kl_loss = 0

    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            recon_data, mu, logvar = model(data)
            loss, recon_loss, kl_loss = vae_loss(recon_data, data, mu, logvar, epoch=1, epochs=1, beta=beta)
            # loss = vae_loss(recon_data, data, mu, logvar, epoch=1, epochs=1, beta=beta)
            test_loss += loss.item()
            test_recon_loss += recon_loss.item()
            test_kl_loss += kl_loss.item()

    avg_loss = test_loss / len(test_loader.dataset)
    avg_recon_loss = test_recon_loss / len(test_loader.dataset)
    avg_kl_loss = test_kl_loss / len(test_loader.dataset)

    print(f'====> Test set loss: {avg_loss:.4f} '
          f'(Recon: {avg_recon_loss:.4f}, KL: {avg_kl_loss:.4f})')

    return avg_loss, avg_recon_loss, avg_kl_loss

def train_VAE(model: VAE, opt: optim.Adam, trainloader: DataLoader, testloader: DataLoader, epochs, beta=1.0):
    train_losses = []
    test_kl_losses = []
    test_recon_losses = []
    test_losses = []

    for epoch in range(1, epochs + 1):
        train_loss, train_recon, train_kl = train_vae_step(model, trainloader, opt, epoch, epochs=epochs+1, beta=beta)
        test_loss, test_recon, test_kl = test_vae_step(model, testloader, beta)

        train_losses.append(train_loss)
        test_kl_losses.append(test_kl)
        test_recon_losses.append(test_recon)
        test_losses.append(test_loss)

    return train_losses, test_losses, test_kl_losses, test_recon_losses

def visualize_gaussian_generations(model, num_samples = 16):
    """Visualize generated samples"""
    import math

    def generate_samples(model, num_samples=64):
        """Generate new samples by sampling from the prior p(z) = N(0, I)"""
        model.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, 128).to(device)
            samples = model.decode(z)
        return samples

    # Reconstructions
    model.eval()
    samples = generate_samples(model, num_samples)

    # Determine grid size automatically (square-ish)
    grid_size = int(math.ceil(math.sqrt(num_samples)))

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size*2, grid_size*2))
    axes = axes.flatten()

    for i in range(num_samples):
        img = samples[i].cpu().permute(1,2,0)
        axes[i].imshow(img)
        axes[i].axis("off")

    # Hide any unused subplots if num_samples is not a perfect square
    for j in range(num_samples, len(axes)):
        axes[j].axis("off")

    plt.suptitle("Generated Samples (Gaussian Prior)")
    plt.tight_layout()
    plt.savefig("VAE_Generations_Base_Training.png", dpi=300, bbox_inches='tight')
    plt.close()

def visualise_reconstructions(model: VAE, dataloader, num_samples=16):
    import math
    model.eval()

    data_iter = iter(dataloader)
    images, _ = next(data_iter)
    images = images[:num_samples].to(device)

    with torch.no_grad():
        recon_images, _, _ = model(images)

    # Determine grid size
    grid_size = int(math.ceil(math.sqrt(num_samples)))

    fig, axes = plt.subplots(grid_size, grid_size*2, figsize=(grid_size*4, grid_size*2))
    axes = axes.flatten()

    for i in range(num_samples):
        # Original image
        img_orig = images[i].cpu().permute(1,2,0)
        axes[2*i].imshow(img_orig)
        axes[2*i].set_title("Original")
        axes[2*i].axis("off")

        # Reconstructed image
        img_recon = recon_images[i].cpu().permute(1,2,0)
        axes[2*i + 1].imshow(img_recon)
        axes[2*i + 1].set_title("Reconstructed")
        axes[2*i + 1].axis("off")

    # Hide any unused axes
    for j in range(2*num_samples, len(axes)):
        axes[j].axis("off")

    plt.suptitle("VAE Reconstructions")
    plt.tight_layout()
    plt.savefig("VAE_Reconstructions.png", dpi=300, bbox_inches='tight')
    plt.close()

def main():
    model = VAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.99))

    print("=== Training VAE ===")
    print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
    print(f"Training on {device}")
    print(f"Epochs: {epochs}")

    train_losses, test_losses, test_kl_losses, test_recon_losses = train_VAE(model=model, opt=optimizer, trainloader=trainloader, testloader=testloader, epochs=epochs, beta=1.0)

    print("=== Plotting Loss Curves ===")
    plt.figure(figsize=(12, 5))

    # Subplot 1: Total Loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss')
    plt.legend()

    # Subplot 2: Recon vs KL (example)
    plt.subplot(1, 2, 2)
    plt.plot(test_recon_losses, label='Test Recon Loss')
    plt.plot(test_kl_losses, label='Test KL Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Recon vs KL Loss (Test)')
    plt.legend()

    plt.tight_layout()
    plt.savefig("VAE_Training_Losses.png", dpi=300, bbox_inches='tight')
    plt.close()

    print("=== Plotting Generated Samples And Reconstructions ===")
    visualize_gaussian_generations(model=model)

    visualise_reconstructions(model=model, dataloader=testloader)

    # Save VAE Model
    torch.save(model.state_dict(), "vae_weights.pth")

if __name__ == "__main__":
    main()