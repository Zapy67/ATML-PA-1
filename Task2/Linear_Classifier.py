import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from sklearn.metrics import classification_report
from torchvision.models import resnet50, ResNet50_Weights
from collections import Counter

import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
    
def inference(model: nn.Module, images:torch.Tensor, class_names: list[str], topk=5, plot=False):
    model.eval()
    with torch.inference_mode():
        outputs = model(images.to(device))
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

def train_step(train_loader: torch.utils.data.DataLoader, model: torch.nn.Module, optimizer: torch.optim.Optimizer, loss_fn: torch.nn.Module, device: torch.device):
    model.to(device), model.train()
    train_loss, num_correct = 0, 0

    for X,Y in train_loader:
        X, Y = X.to(device), Y.to(device)
        optimizer.zero_grad()
        preds = model(X)
        loss = loss_fn(preds,Y)
        num_correct += sum(preds.argmax(axis=1)==Y).item()
        loss.backward()
        optimizer.step()
        train_loss+=loss.item()

    train_loss/=len(train_loader)
    acc = num_correct/(len(train_loader.dataset))
    print(f"\nTrain loss: {train_loss:.5f} |  Train acc: {acc*100:.2f} %\n", flush=True)
    return train_loss, acc

def test_step(test_loader: torch.utils.data.DataLoader, model: torch.nn.Module, loss_fn: torch.nn.Module, device: torch.device):
    model.to(device), model.eval()
    test_loss, num_correct = 0, 0

    with torch.inference_mode():
        for X,Y in test_loader:
            X, Y = X.to(device), Y.to(device)
            preds = model(X)
            test_loss += loss_fn(preds,Y).item()
            num_correct += sum(preds.argmax(axis=1)==Y).item()

    test_loss /= len(test_loader)
    acc = num_correct/len(test_loader.dataset)
    print(f"Test loss: {test_loss:.5f} | Test acc: {acc*100:.2f} %\n", flush=True)
    return test_loss, acc

def train(train_loader: torch.utils.data.DataLoader, test_loader: torch.utils.data.DataLoader, epochs: int, optimizer: torch.optim.Optimizer, model: torch.nn.Module, loss_fn: torch.nn.Module, device: torch.device):
    model.to(device)
    train_losses, test_losses, train_accuracies, test_accuracies = [], [], [], []
   
    for epoch in tqdm.trange(epochs):
        print(f"epoch: {epoch+1}")
        train_loss, train_acc = train_step(train_loader, model, optimizer, loss_fn, device)
        test_loss, test_acc = test_step(test_loader, model, loss_fn, device)
        train_losses.append(train_loss), train_accuracies.append(train_acc)
        test_losses.append(test_loss), test_accuracies.append(test_acc)

    return train_losses, test_losses, train_accuracies, test_accuracies

def plot_dynamics(train_loss_hist, test_loss_hist, train_acc_hist, test_acc_hist):
    fig, axes= plt.subplots(1,2)
    fig.suptitle("Training Dynamics")
    fig.set_size_inches(10, 3)
    ax1, ax2 = axes

    ax1.set_title("Accuracies")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.plot(train_acc_hist, label="Train")
    ax1.plot(test_acc_hist, label="Test")
    ax1.legend()
    ax1.grid(True)

    ax2.set_title("Losses")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.plot(train_loss_hist, label="Train")
    ax2.plot(test_loss_hist, label="Test")
    ax2.legend()
    ax2.grid(True)
    
def prep_dataset():
    dataset = CIFAR10(root='./data', train=True, download=True, transform=ResNet50_Weights.IMAGENET1K_V2.transforms())
    trainset, testset = torch.utils.data.random_split(dataset, [0.5, 0.5])
    classes = dataset.classes
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=os.cpu_count())
    test_loader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=os.cpu_count())

    return train_loader, test_loader, classes

def train_model(save: bool = True):
    loss_fn = torch.nn.CrossEntropyLoss()

    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    for p in model.parameters():
        p.requires_grad = False
    
    train_loader, test_loader, classes = prep_dataset()

    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model = model.to(device)
    optimizer = torch.optim.SGD(model.fc.parameters(), lr=1e-3, weight_decay=1e-5)
    num_params = sum([p.numel() for p in model.fc.parameters() if p.requires_grad])

    print(f"=== Num Params: {num_params} ===")

    epochs = 10

    train_losses, test_losses, train_accuracies, test_accuracies=train(train_loader,
                        test_loader,epochs, optimizer,  model, loss_fn, device)
    
    plot_dynamics(train_losses, test_losses, train_accuracies, test_accuracies)

    all_preds, all_labels = [], []
    model.eval()
    with torch.inference_mode():
        for X, Y in test_loader:
            X, Y = X.to(device), Y.to(device)
            preds = model(X).argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(Y.cpu().numpy())

    print("\n=== Classification Report ===")
    print(classification_report(all_labels, all_preds, target_names=classes))

    if save:
        torch.save(model.fc.state_dict(), "lin_classifier_head.pth")
        return classes
    else:
        return model, classes