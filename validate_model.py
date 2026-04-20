import os
import json
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import random
from tqdm import tqdm

# Config
MODEL_PATH = "model/leaf_disease_model.pt"
LABELS_PATH = "model/class_labels.json"
DATA_DIR = "data/train"
RESULTS_PATH = "test_results.json"
IMG_SIZE = 224
N_TTA = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Labels
with open(LABELS_PATH, "r") as f:
    class_labels = json.load(f)

# Load Model
def get_model(num_classes):
    model = models.mobilenet_v2(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, num_classes)
    )
    return model

model = get_model(len(class_labels))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

tta_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

results = []

# Collect test images (sample up to 10 per class)
test_images = []
for label in class_labels:
    class_path = os.path.join(DATA_DIR, label)
    if os.path.exists(class_path):
        imgs = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        sampled = random.sample(imgs, min(len(imgs), 10))
        for img_name in sampled:
            test_images.append((os.path.join(class_path, img_name), label))

print(f"Running validation on {len(test_images)} images...")

for img_path, expected in tqdm(test_images):
    try:
        image = Image.open(img_path).convert("RGB")
        
        # Inference
        anchor_tensor = transform(image).unsqueeze(0).to(device)
        tta_tensors = [tta_transform(image).to(device) for _ in range(N_TTA)]
        batch_tensor = torch.cat([anchor_tensor, torch.stack(tta_tensors)], dim=0)
        
        with torch.no_grad():
            outputs = model(batch_tensor)
            probabilities = torch.nn.functional.softmax(outputs / 1.5, dim=1) # Temperature scaling
            
            mean_probs = torch.mean(probabilities, dim=0)
            confidence, predicted_idx = torch.max(mean_probs, 0)
            
            predicted_classes = torch.argmax(probabilities, dim=1)
            consistency_score = (predicted_classes == predicted_idx.item()).sum().item() / (N_TTA + 1)
            
            top3_prob, top3_idx = torch.topk(mean_probs, min(3, len(class_labels)))
            
            confidence_val = confidence.item() * 100
            top2_val = top3_prob[1].item() * 100 if len(top3_prob) > 1 else 0.0
            margin = confidence_val - top2_val
            epsilon = 1e-7
            entropy = -torch.sum(mean_probs * torch.log(mean_probs + epsilon)).item()
            
            predicted_label = class_labels[predicted_idx.item()]
            
            results.append({
                "path": img_path,
                "expected": expected,
                "predicted": predicted_label,
                "confidence": round(confidence_val, 2),
                "margin": round(margin, 2),
                "entropy": round(entropy, 4),
                "consistency": round(consistency_score, 2),
                "is_correct": expected == predicted_label
            })
    except Exception as e:
        print(f"Error processing {img_path}: {e}")

with open(RESULTS_PATH, "w") as f:
    json.dump(results, f, indent=4)

# Simple Confusion Matrix Logic
from collections import defaultdict
matrix = defaultdict(lambda: defaultdict(int))
for res in results:
    matrix[res["expected"]][res["predicted"]] += 1

print("\n==== CONFUSION MATRIX ====")
header = "EXP \ PRED | " + " | ".join([l[:10] for l in class_labels])
print(header)
print("-" * len(header))
for exp in class_labels:
    row = f"{exp[:10]:10} | "
    for pred in class_labels:
        row += f"{matrix[exp][pred]:10} | "
    print(row)

accuracy = sum(1 for r in results if r["is_correct"]) / len(results) if results else 0
print(f"\nOverall Validation Accuracy: {accuracy*100:.2f}%")
print(f"Results saved to {RESULTS_PATH}")
