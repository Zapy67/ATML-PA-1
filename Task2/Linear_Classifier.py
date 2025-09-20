import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"

CIFAR_10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]

class CIFAR10Classifier(nn.Module):
    def __init__(self, classes=10):
        super(CIFAR10Classifier, self).__init__()

        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Dropout(0.25)
            )

        self.backbone = nn.Sequential(
            conv_block(3, 64),     # 32x32 -> 16x16
            conv_block(64, 128),   # 16x16 -> 8x8
            conv_block(128, 256),  # 8x8 -> 4x4
        )

        self.fc = nn.Linear(256 * 4 * 4, classes)

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)           # (Batch, classes)
        return x
    
    def inference(self, images, class_names, topk=5, plot=False):
        self.eval()
        with torch.no_grad():
            device = next(self.parameters()).device
            outputs = self(images.to(device))
            probs = F.softmax(outputs, dim=1)
            top_probs, top_idxs = probs.topk(topk, dim=1)

        preds_topk = []
        top1_preds = []
        for idxs, probs_row in zip(top_idxs, top_probs):
            row_preds = [(class_names[i.item()], float(p)) for i, p in zip(idxs, probs_row)]
            preds_topk.append(row_preds)
            top1_preds.append(row_preds[0][0])

        if plot:
            counts = Counter(top1_preds)
            plt.figure(figsize=(8, 4))
            plt.bar(counts.keys(), counts.values())
            plt.xticks(rotation=45)
            plt.xlabel("Class")
            plt.ylabel("Frequency (Top-1 Predictions)")
            plt.title("Distribution of Top-1 Predictions")
            plt.tight_layout()
            plt.show()

        return preds_topk

def train(model: CIFAR10Classifier, trainloader, testloader, epochs=40, lr=1e-3):
    opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()
    model = model.to(device)
    train_losses, test_losses = [], []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        correct, total = 0, 0
        for imgs, labels in trainloader:
            imgs, labels = imgs.to(device), labels.to(device)

            opt.zero_grad()
            out = model(imgs)
            loss = loss_fn(out, labels)
            loss.backward()
            opt.step()
            running_loss += loss.item() * imgs.size(0)
            preds = out.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / len(trainloader.dataset)
        train_losses.append(epoch_loss)
        train_acc = 100.0 * correct / total

        model.eval()
        correct, total, test_loss = 0, 0, 0.0
        with torch.no_grad():
            for imgs, labels in testloader:
                imgs, labels = imgs.to(device), labels.to(device)
                out = model(imgs)
                loss = loss_fn(out, labels)
                test_loss += loss.item() * imgs.size(0)

                preds = out.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        test_loss /= len(testloader.dataset)
        test_losses.append(test_loss)

        test_acc = 100.0 * correct / total

        print(f"Epoch [{epoch+1}/{epochs}] Loss: {epoch_loss:.4f}, Train Acc:{train_acc:.2f}, Test Acc: {test_acc:.2f}%")

    return model, train_losses, test_losses

def prep_dataset():
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])

    trainset = CIFAR10(root="./data", train=True, download=True, transform=transform_train)
    testset = CIFAR10(root="./data", train=False, download=True, transform=transform_test)

    trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
    testloader = DataLoader(testset, batch_size=64, shuffle=False)

    return trainloader, testloader

def main():
    print(f"Device = {device}")
    print("Prepping Dataset")
    trainloader, testloader = prep_dataset()

    model = CIFAR10Classifier()
    print("Training Model")
    model, train_losses, test_losses = train(model, trainloader, testloader, epochs=20, lr=0.01)

    # Plot Losses
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(test_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training vs Testing Loss")
    plt.savefig("loss_curve_classifier.png")

    torch.save(model.state_dict(), "classifier.pth")
    print("Model weights saved to classifier.pth")

if __name__ == "__main__":
    main()