import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from PIL import Image
import torchvision.transforms as transforms

class MNISTDataset(Dataset):
    def __init__(self, csv_file, train=True, transform=None):
        self.data = pd.read_csv(csv_file)
        self.train = train
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.train:
            label = self.data.iloc[idx, 0]
            pixels = self.data.iloc[idx, 1:].values.astype(np.float32) / 255.0
            img = pixels.reshape(28, 28)
            img = Image.fromarray(img.astype(np.uint8), mode='L')
            if self.transform:
                img = self.transform(img)
            return img, torch.tensor(label, dtype=torch.long)
        else:
            pixels = self.data.iloc[idx, :].values.astype(np.float32) / 255.0
            img = pixels.reshape(28, 28)
            img = Image.fromarray(img.astype(np.uint8), mode='L')
            if self.transform:
                img = self.transform(img)
            return img

class AdvancedCNN(nn.Module):
    def __init__(self):
        super(AdvancedCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25)
        )
        
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    return running_loss / len(train_loader), 100 * correct / total

def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return running_loss / len(val_loader), 100 * correct / total

def predict(model, test_loader, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for images in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            predictions.extend(predicted.cpu().numpy())
    return predictions

def main():
    print("="*60)
    print("Optimized CNN Training for 99%+ Accuracy")
    print("="*60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_transform = transforms.Compose([
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=10, translate=(0.15, 0.15)),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    full_dataset = MNISTDataset('digit-recognizer/train.csv', train=True, transform=train_transform)
    test_dataset = MNISTDataset('digit-recognizer/test.csv', train=False, transform=test_transform)

    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

    model = AdvancedCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

    best_val_acc = 0.0
    best_model_state = None
    epochs_no_improve = 0
    early_stop_patience = 5

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    epochs = 30

    print("\nTraining started...")
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            epochs_no_improve = 0
            print(f"✓ New best! Val Acc: {val_acc:.4f}%")
        else:
            epochs_no_improve += 1

        print(f"Epoch [{epoch+1}/{epochs}] LR: {scheduler.get_last_lr()[0]:.6f} | "
              f"Train Loss: {train_loss:.4f} Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} Val Acc: {val_acc:.2f}%")

        if epochs_no_improve >= early_stop_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    model.load_state_dict(best_model_state)

    print("\n" + "="*60)
    print(f"FINAL RESULTS")
    print("="*60)
    print(f"Best Val Accuracy: {best_val_acc:.4f}%")
    print(f"Final Train Accuracy: {train_accs[-1]:.4f}%")

    predictions = predict(model, test_loader, device)
    submission = pd.DataFrame({
        'ImageId': range(1, len(predictions)+1),
        'Label': predictions
    })
    submission.to_csv('digit-recognizer/sample_submission.csv', index=False)
    submission.to_csv('digit-recognizer/submission_Final.csv', index=False)

    torch.save(model.state_dict(), 'cnn_model.pth')

    plt.figure(figsize=(14, 10))
    plt.plot(train_losses, label='Train Loss', color='#1f77b4', linewidth=2.5)
    plt.plot(val_losses, label='Val Loss', color='#ff7f0e', linewidth=2.5)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.title('Training and Validation Loss (99%+ Accuracy)', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('digit-recognizer/loss_curves.png', dpi=200, bbox_inches='tight')
    plt.close()

    print("\nFiles saved:")
    print("- digit-recognizer/sample_submission.csv")
    print("- cnn_model.pth")
    print("- digit-recognizer/loss_curves.png")

    return {
        'train_acc': train_accs[-1],
        'val_acc': best_val_acc,
        'epochs': len(train_losses),
        'min_loss': min(train_losses)
    }

if __name__ == '__main__':
    results = main()
    print(f"\n✅ Achieved {results['val_acc']:.2f}% accuracy!")