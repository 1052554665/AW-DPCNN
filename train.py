# The script to train the AlexNet model on the dataset, log training and validation metrics
# to TensorBoard, and save the trained model and metrics history to files.
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import AlexNet
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

train_dataset = datasets.ImageFolder(root='./datasets/train', transform=train_transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=16)

val_dataset = datasets.ImageFolder(root='./datasets/val', transform=val_transform)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=16)

model = AlexNet(num_classes=len(train_dataset.classes)).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), weight_decay=0.0005, lr=0.001, momentum=0.9)

writer = SummaryWriter('runs/alexnet_experiment')

metrics = {
    'train_loss': [],
    'val_loss': [],
    'train_acc': [],
    'val_acc': []
}

def train_model(model, criterion, optimizer, num_epochs=50):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total

        metrics['train_loss'].append(epoch_loss)
        metrics['train_acc'].append(epoch_acc)

        writer.add_scalar('Loss/train', epoch_loss, epoch)
        writer.add_scalar('Accuracy/train', epoch_acc, epoch)
        print(f'Epoch {epoch + 1}/{num_epochs} Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        epoch_val_loss = val_loss / len(val_loader)
        epoch_val_acc = correct / total

        metrics['val_loss'].append(epoch_val_loss)
        metrics['val_acc'].append(epoch_val_acc)

        writer.add_scalar('Loss/validation', epoch_val_loss, epoch)
        writer.add_scalar('Accuracy/validation', epoch_val_acc, epoch)

        print(f'Epoch {epoch + 1}/{num_epochs} Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f}')

if __name__ == '__main__':
    print("using {} device.".format(device))
    train_model(model, criterion, optimizer, num_epochs=20)
    with open('metrics_history.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    writer.close()

    def save_model(model, model_path):
        torch.save(model.state_dict(), model_path)

    model_save_path = './alexnet.pth'
    save_model(model, model_save_path)
    print(f'Model saved to {model_save_path}')