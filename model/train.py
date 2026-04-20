import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import time
from collections import Counter

# Configurations
DATA_DIR = "../data/train"
VAL_DIR = "../data/val"
MODEL_SAVE_PATH = "leaf_disease_model.pt"
LABELS_SAVE_PATH = "class_labels.json"
BATCH_SIZE = 32
EPOCHS = 25
IMG_SIZE = 224

def get_model(num_classes):
    print("Loading MobileNetV2 with Advanced Transfer Learning...")
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, num_classes)
    )
    return model

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if not os.path.exists(DATA_DIR) or len(os.listdir(DATA_DIR)) == 0:
        print("CRITICAL ERROR: No real data found.")
        return

    full_dataset = datasets.ImageFolder(DATA_DIR, transform=train_transform)
    class_names = full_dataset.classes
    with open(LABELS_SAVE_PATH, "w") as f:
        json.dump(class_names, f)
    
    # CALCULATE CLASS WEIGHTS for balancing
    targets = [s[1] for s in full_dataset.samples]
    counts = Counter(targets)
    cls_counts = [counts[i] for i in range(len(class_names))]
    total = sum(cls_counts)
    # Inverse frequency weighting
    weights = [total / (len(class_names) * c) if c > 0 else 0 for c in cls_counts]
    class_weights = torch.FloatTensor(weights).to(device)
    print(f"Applying Class Weights: {weights}")

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    val_dataset.dataset.transform = val_transform # Use val_transform for val split

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = get_model(len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.0005)

    print(f"Starting Balanced training loop...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        start_time = time.time()
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            
        epoch_loss = running_loss / len(train_dataset)
        
        model.eval()
        correct = 0
        total_val = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        val_acc = 100 * correct / total_val
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {epoch_loss:.4f} - Val Acc: {val_acc:.2f}% - Time: {time.time() - start_time:.2f}s")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Balanced Model saved.")

if __name__ == "__main__":
    train_model()
