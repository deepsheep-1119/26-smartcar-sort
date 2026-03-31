import torch
import torch.nn as nn
from torchvision import transforms
from pathlib import Path
import cv2
import numpy as np


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


def predict_image(model, img_path, idx_to_class, device):
    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((96, 96)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Failed to load {img_path}")
        return None

    img_tensor = transform(img).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        output = model(img_tensor)
        pred = output.argmax(dim=1).item()

    return idx_to_class[pred]


def main():
    checkpoint = torch.load("smartcar_model.pth", weights_only=False)
    idx_to_class = checkpoint["idx_to_class"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN(num_classes=3).to(device)
    model.load_state_dict(checkpoint["model"])

    print(f"Using device: {device}")
    print(f"Classes: {idx_to_class}")

    test_dir = Path("data/smartcar/test")
    categories = ["交通工具-直行", "武器-左", "物资-右"]

    correct = 0
    total = 0

    for cat in categories:
        cat_dir = test_dir / cat
        if not cat_dir.exists():
            continue
        for img_path in cat_dir.glob("*.png"):
            pred = predict_image(model, img_path, idx_to_class, device)
            true_label = cat
            is_correct = pred == true_label
            correct += is_correct
            total += 1
            status = "✓" if is_correct else "✗"
            print(f"{status} {img_path.name}: predicted={pred}, actual={true_label}")

    print(f"\nAccuracy: {correct}/{total} = {100 * correct / total:.2f}%")


if __name__ == "__main__":
    main()
