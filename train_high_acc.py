import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
            img = Image.fromarray((img * 255).astype(np.uint8), mode='L')
            if self.transform:
                img = self.transform(img)
            return img, torch.tensor(label, dtype=torch.long)
        else:
            pixels = self.data.iloc[idx, :].values.astype(np.float32) / 255.0
            img = pixels.reshape(28, 28)
            img = Image.fromarray((img * 255).astype(np.uint8), mode='L')
            if self.transform:
                img = self.transform(img)
            return img

class HighAccCNN(nn.Module):
    def __init__(self):
        super(HighAccCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 10)
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
    print("Quick Training High Accuracy CNN Model")
    print("Target: Validation Accuracy >= 99%")
    print("="*60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)

    train_transform = transforms.Compose([
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=10, translate=(0.15, 0.15)),
        transforms.RandomPerspective(distortion_scale=0.1),
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

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=0)

    model = HighAccCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)

    best_val_acc = 0.0
    best_model_state = None

    print("\nTraining started...")
    for epoch in range(15):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()

        print("Epoch [{}/15] Train Loss: {:.4f} Train Acc: {:.2f}% | Val Loss: {:.4f} Val Acc: {:.2f}%".format(
            epoch+1, train_loss, train_acc, val_loss, val_acc))

        if val_acc >= 99.0:
            print("Achieved target accuracy {}%!".format(val_acc))
            break

    model.load_state_dict(best_model_state)

    print("\nTraining completed!")
    print("Best Validation Accuracy: {:.2f}%".format(best_val_acc))

    print("\nGenerating Kaggle submission...")
    predictions = predict(model, test_loader, device)
    submission = pd.DataFrame({
        'ImageId': range(1, len(predictions)+1),
        'Label': predictions
    })
    submission.to_csv('digit-recognizer/sample_submission.csv', index=False)

    torch.save(model.state_dict(), 'cnn_model.pth')
    print("\nFiles saved:")
    print("  - cnn_model.pth")
    print("  - digit-recognizer/sample_submission.csv")

    if best_val_acc >= 99.0:
        print("\nSUCCESS: Accuracy reached 99%+!")
    else:
        print("Current accuracy {:.2f}%".format(best_val_acc))

if __name__ == '__main__':
    main()