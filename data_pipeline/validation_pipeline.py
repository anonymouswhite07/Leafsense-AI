import os
import io
import json
import shutil
import logging
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')

# Configurations
TRAIN_DIR = "../data/train"
REVIEW_DIR = "../data/review"
UNKNOWN_DIR = "../data/train/Unknown"
MODEL_DIR = "../model"
LABELS_PATH = os.path.join(MODEL_DIR, "class_labels.json")
ACTIVE_MODEL_PATH = os.path.join(MODEL_DIR, "leaf_disease_model.pt")
REPORT_PATH = "../data/quality_report.json"

SUSPICIOUS_THRESHOLD = 0.85 # Flag if model predicts different class with > 85% confidence

class DatasetValidator:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info("Loading Models...")
        
        # 1. Load CLIP for relevance
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_labels = ["a photo of a plant leaf", "a photo of a crop disease", "a person", "a logo or text", "a diagram", "an animal or bug"]
        
        # 2. Load Active Model for verification
        self.labels = []
        if os.path.exists(LABELS_PATH):
             with open(LABELS_PATH, "r") as f:
                 self.labels = json.load(f)
                 
        if not self.labels:
             logging.error("No class labels found! Please run clean_dataset.py and train.py first.")
             return
             
        self.classifier = models.mobilenet_v2(weights=None)
        in_features = self.classifier.classifier[1].in_features
        self.classifier.classifier[1] = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, len(self.labels))
        )
        if os.path.exists(ACTIVE_MODEL_PATH):
             self.classifier.load_state_dict(torch.load(ACTIVE_MODEL_PATH, map_location=self.device))
        self.classifier.to(self.device)
        self.classifier.eval()
        
        self.cls_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def filter_relevance(self, img):
        # FEATURE 1: IMAGE RELEVANCE FILTER via CLIP
        inputs = self.clip_processor(text=self.clip_labels, images=img, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
             outputs = self.clip_model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)[0]
        best_idx = probs.argmax().item()
        # Indices 0 and 1 are plant/crop related. Others are irrelevant
        return best_idx in [0, 1]

    def verify_label(self, img, actual_label):
        # FEATURE 2: SEMI-AUTOMATIC VERIFICATION
        tensor = self.cls_transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.classifier(tensor)
            probs = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, pre_idx = torch.max(probs, 0)
        
        pred_label = self.labels[pre_idx.item()]
        
        if pred_label != actual_label and confidence.item() > SUSPICIOUS_THRESHOLD:
            return False, pred_label, confidence.item()
        return True, pred_label, confidence.item()

    def run_pipeline(self):
        os.makedirs(REVIEW_DIR, exist_ok=True)
        os.makedirs(UNKNOWN_DIR, exist_ok=True) # FEATURE 5: HARD NEGATIVE SET
        
        report = {
            "total_scanned": 0,
            "irrelevant_discarded": 0,
            "flagged_suspicious": 0,
            "verified_images": 0
        }
        
        classes = [d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))]
        
        for cls in classes:
            if cls == "Unknown": continue
            
            cls_path = os.path.join(TRAIN_DIR, cls)
            files = os.listdir(cls_path)
            
            for file in tqdm(files, desc=f"Validating {cls}", leave=False):
                report["total_scanned"] += 1
                filepath = os.path.join(cls_path, file)
                
                try:
                    img = Image.open(filepath).convert("RGB")
                    
                    # 1. Check Relevance
                    is_relevant = self.filter_relevance(img)
                    if not is_relevant:
                        # Move to Unknown set as a hard negative or delete completely. We'll use Feature 5.
                        shutil.move(filepath, os.path.join(UNKNOWN_DIR, f"hard_neg_{file}"))
                        report["irrelevant_discarded"] += 1
                        continue
                        
                    # 2. Verify Label
                    is_correct, pred_label, conf = self.verify_label(img, cls)
                    if not is_correct:
                        # Move to review
                        review_path = os.path.join(REVIEW_DIR, f"{cls}___pred_{pred_label}___{conf:.2f}___{file}")
                        shutil.move(filepath, review_path)
                        report["flagged_suspicious"] += 1
                    else:
                        report["verified_images"] += 1
                        
                except Exception as e:
                    logging.warning(f"Error processing {file}: {str(e)}")
                    
        # Feature 4: Quality Score Calculation
        report["dataset_quality_score"] = report["verified_images"] / max(1, report["total_scanned"]) * 100
        with open(REPORT_PATH, "w") as f:
            json.dump(report, f, indent=4)
            
        logging.info(f"Pipeline Complete. Quality Report: {json.dumps(report, indent=2)}")
        logging.info(f"Please check {REVIEW_DIR} folder using review_cli.py")

if __name__ == "__main__":
    validator = DatasetValidator()
    validator.run_pipeline()
