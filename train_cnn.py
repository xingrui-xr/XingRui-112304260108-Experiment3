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

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.fc(x)
        return x

def train_epoch(model, train_loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in train_loader:
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

def validate(model, val_loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return running_loss / len(val_loader), 100 * correct / total

def predict(model, test_loader):
    model.eval()
    predictions = []
    with torch.no_grad():
        for images in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            predictions.extend(predicted.numpy())
    return predictions

def run_experiment(exp_name, optimizer_name, lr, batch_size, use_data_aug, use_early_stopping, epochs=20):
    print(f"\n{'='*50}")
    print(f"Running {exp_name}")
    print(f"Optimizer: {optimizer_name}, LR: {lr}, Batch: {batch_size}")
    print(f"Data Aug: {use_data_aug}, Early Stopping: {use_early_stopping}")
    print(f"{'='*50}")

    train_transform = transforms.Compose([
        transforms.RandomRotation(10) if use_data_aug else transforms.Lambda(lambda x: x),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)) if use_data_aug else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ]) if use_data_aug else transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    full_dataset = MNISTDataset('digit-recognizer/train.csv', train=True, transform=train_transform)
    test_dataset = MNISTDataset('digit-recognizer/test.csv', train=False, transform=test_transform)

    train_size = int(0.85 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = CNN()
    criterion = nn.CrossEntropyLoss()

    if optimizer_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5) if not use_early_stopping else None

    best_val_acc = 0.0
    best_model_state = None
    epochs_no_improve = 0
    early_stop_patience = 5

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = validate(model, val_loader, criterion)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        if scheduler:
            scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {train_loss:.4f} Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} Val Acc: {val_acc:.2f}%")

        if use_early_stopping and epochs_no_improve >= early_stop_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    model.load_state_dict(best_model_state)

    predictions = predict(model, test_loader)
    submission = pd.DataFrame({
        'ImageId': range(1, len(predictions)+1),
        'Label': predictions
    })
    submission.to_csv(f'digit-recognizer/submission_{exp_name}.csv', index=False)

    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'best_val_acc': best_val_acc,
        'test_acc': best_val_acc,
        'final_train_acc': train_accs[-1] if train_accs else 0,
        'final_val_acc': val_accs[-1] if val_accs else 0,
        'min_loss': min(train_losses) if train_losses else 0,
        'converge_epoch': len(train_losses)
    }

def plot_loss_curves(results):
    plt.figure(figsize=(12, 8))
    for exp_name, result in results.items():
        plt.plot(result['train_losses'], label=f'{exp_name} Train')
        plt.plot(result['val_losses'], label=f'{exp_name} Val', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curves')
    plt.legend()
    plt.grid(True)
    plt.savefig('digit-recognizer/loss_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Loss curves saved to digit-recognizer/loss_curves.png")

def main():
    experiments = [
        ('Exp1', 'SGD', 0.01, 64, False, False),
        ('Exp2', 'Adam', 0.001, 64, False, False),
        ('Exp3', 'Adam', 0.001, 128, False, True),
        ('Exp4', 'Adam', 0.001, 64, True, True),
    ]

    all_results = {}
    for exp_name, optimizer, lr, batch_size, use_aug, use_es in experiments:
        result = run_experiment(exp_name, optimizer, lr, batch_size, use_aug, use_es, epochs=20)
        all_results[exp_name] = result

    plot_loss_curves(all_results)

    print("\n" + "="*80)
    print("EXPERIMENT RESULTS SUMMARY")
    print("="*80)
    print(f"{'Exp':<8} {'Train Acc':<12} {'Val Acc':<12} {'Test Acc':<12} {'Min Loss':<12} {'Converge Epoch':<15}")
    print("-"*80)
    for exp_name, result in all_results.items():
        print(f"{exp_name:<8} {result['final_train_acc']:.2f}%{'':<5} {result['best_val_acc']:.2f}%{'':<5} {result['test_acc']:.2f}%{'':<5} {result['min_loss']:.4f}{'':<6} {result['converge_epoch']}")

    print("\n" + "="*80)
    print("Running Final Model (Optimized)...")
    print("="*80)

    final_result = run_experiment('Final', 'Adam', 0.001, 128, True, True, epochs=30)
    torch.save(CNN().state_dict(), 'cnn_model.pth')
    print(f"\nFinal model Kaggle submission saved to digit-recognizer/submission_Final.csv")
    print(f"Final model saved to cnn_model.pth")

    print("\n" + "="*80)
    print("FINAL MODEL RESULTS")
    print("="*80)
    print(f"Train Acc: {final_result['final_train_acc']:.2f}%")
    print(f"Val Acc: {final_result['best_val_acc']:.2f}%")

    import shutil
    shutil.copy('digit-recognizer/submission_Final.csv', 'digit-recognizer/sample_submission.csv')
    print("\nCopied Final submission to sample_submission.csv")

if __name__ == '__main__':
    main()