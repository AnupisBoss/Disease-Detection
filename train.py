import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models.cnn_model import SimpleCNN
from utils import calculate_accuracy
from torch.utils.data import Subset

# Use only first 1000 samples for quick training


# Hyperparams
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001
NUM_CLASSES = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data transforms (resize + normalization)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],   # Imagenet means
                         [0.229, 0.224, 0.225])   # Imagenet stds
])

# Datasets
train_data = datasets.ImageFolder('data/train', transform=transform)
test_data = datasets.ImageFolder('data/test', transform=transform)

# DataLoaders
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

# Model, loss, optimizer
model = SimpleCNN(num_classes=NUM_CLASSES).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

def train():
    model.train()
    running_loss = 0
    running_acc = 0
    for i, (images, labels) in enumerate(train_loader):
            print(f"Processing batch {i + 1} / {len(train_loader)}")

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_acc += calculate_accuracy(outputs, labels)

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = running_acc / len(train_loader)
    print(f"Train Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

def evaluate():
    model.eval()
    running_acc = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            running_acc += calculate_accuracy(outputs, labels)

    epoch_acc = running_acc / len(test_loader)
    print(f"Test Accuracy: {epoch_acc:.4f}")

if __name__ == "__main__":
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}")
        train()
        evaluate()

    torch.save(model.state_dict(), 'models/cnn_model.pth')
    print("Model saved to models/cnn_model.pth")
