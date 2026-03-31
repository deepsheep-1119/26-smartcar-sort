import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class CNN(nn.Module):
    def __init__(self, num_classes=3):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 12 * 12, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 128 * 12 * 12)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


def get_dataLoaders(batch_size=16, img_size=96):
    transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    CLASSES = ["交通工具-直行", "武器-左", "物资-右"]
    IDX_TO_CLASS = {i: c for i, c in enumerate(CLASSES)}

    train_dataset = datasets.ImageFolder(
        root="data/smartcar/train", transform=transform
    )
    test_dataset = datasets.ImageFolder(root="data/smartcar/test", transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, IDX_TO_CLASS


def train(epochs=20):
    train_loader, test_loader, idx_to_class = get_dataLoaders()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN(num_classes=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(f"Using device: {device}")
    print(f"Classes: {idx_to_class}")
    print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()

        accuracy = 100.0 * correct / len(test_loader.dataset)
        print(
            f"Epoch {epoch + 1}/{epochs} - Loss: {total_loss / len(train_loader):.4f} - Acc: {accuracy:.2f}%"
        )

    torch.save(
        {"model": model.state_dict(), "idx_to_class": idx_to_class},
        "smartcar_model.pth",
    )
    print("Model saved to smartcar_model.pth")


if __name__ == "__main__":
    train()
