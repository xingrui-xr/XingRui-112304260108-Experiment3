import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR
import pandas as pd
import numpy as np

class MNISTDataset(Dataset):
    def __init__(self, csv_file, train=True):
        self.data = pd.read_csv(csv_file)
        self.train = train
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.train:
            label = self.data.iloc[idx, 0]
            pixels = self.data.iloc[idx, 1:].values.astype(np.float32) / 255.0
            return torch.tensor(pixels), torch.tensor(label, dtype=torch.long)
        else:
            pixels = self.data.iloc[idx, :].values.astype(np.float32) / 255.0
            return torch.tensor(pixels)

class ImprovedDNN(nn.Module):
    def __init__(self, input_size=784, num_classes=10):
        super(ImprovedDNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.3)
        
        self.fc4 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout1(out)
        
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        
        out = self.fc3(out)
        out = self.bn3(out)
        out = self.relu3(out)
        out = self.dropout3(out)
        
        out = self.fc4(out)
        return out

def train(model, train_loader, criterion, optimizer, scheduler, num_epochs=30):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for i, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        scheduler.step()
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')

def main():
    train_dataset = MNISTDataset('digit-recognizer/train.csv', train=True)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    
    model = ImprovedDNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    
    print('Training started...')
    train(model, train_loader, criterion, optimizer, scheduler, num_epochs=30)
    
    torch.save(model.state_dict(), 'digit_model.pth')
    print('Model saved to digit_model.pth')

if __name__ == '__main__':
    main()