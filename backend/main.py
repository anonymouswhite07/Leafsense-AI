import os
import json
from dotenv import load_dotenv
load_dotenv()
import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
import cv2
import base64
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import logging
import sqlite3
import time
from typing import List, Optional
from pydantic import BaseModel
import google.generativeai as genai

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")

import gc
from starlette.middleware.base import BaseHTTPMiddleware

app = FastAPI(title="LeafSense AI API")

# Global Error + CORS Handler (Ensures CORS headers even on crashes)
class CORSRecoveryMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        try:
            response = await call_next(request)
            response.headers["Access-Control-Allow-Origin"] = "*"
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
            response.headers["Access-Control-Allow-Headers"] = "*"
            return response
        except Exception as e:
            logger.error(f"CRITICAL SYSTEM ERROR: {e}")
            return JSONResponse(
                status_code=500,
                content={"error": "Internal server error. RAM limit likely exceeded on Free Tier."},
                headers={
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Methods": "*",
                    "Access-Control-Allow-Headers": "*"
                }
            )

app.add_middleware(CORSRecoveryMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# TTA Configuration (Reduced for RAM stability)
TTA_ITERATIONS = 5 

# Constants & Configurations
MODEL_DIR = "model" if os.path.exists("model") else "../model"
MODEL_PATH = os.path.join(MODEL_DIR, "leaf_disease_model.pt")
LABELS_PATH = os.path.join(MODEL_DIR, "class_labels.json")
SOLUTIONS_PATH = "solutions.json"
DATABASE_PATH = "intelligence.db"

# Load Labels & Solutions
with open(LABELS_PATH, "r") as f:
    class_labels = json.load(f)

with open(SOLUTIONS_PATH, "r") as f:
    solutions = json.load(f)

# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model Loading
def get_model(num_classes):
    model = models.mobilenet_v2(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, num_classes)
    )
    return model

model = get_model(len(class_labels))
if os.path.exists(MODEL_PATH):
    logger.info("Loading PyTorch model...")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# Grad-CAM Implementation
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor, class_idx):
        self.model.zero_grad()
        output = self.model(input_tensor)
        loss = output[0, class_idx]
        loss.backward()
        
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1).squeeze().detach().cpu().numpy()
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam) if np.max(cam) != 0 else cam
        return cam

grad_cam = GradCAM(model, model.features[18])

# Database & Outbreak Logic
def get_db():
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            disease TEXT,
            confidence REAL,
            timestamp TEXT,
            lat REAL,
            lng REAL,
            region TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

def analyze_outbreaks(region):
    conn = get_db()
    c = conn.cursor()
    # Check for spike in specific disease in region in last 24h
    c.execute("SELECT disease, COUNT(*) as count FROM events WHERE region = ? GROUP BY disease", (region,))
    stats = c.fetchall()
    conn.close()
    
    threshold = 3
    alerts = []
    for row in stats:
        if row['count'] >= threshold:
            alerts.append(f"Potential outbreak detected: {row['disease'].replace('___', ' ')} in {region}!")
    return alerts

# Image Transforms
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

@app.post("/predict")
async def predict(
    files: List[UploadFile] = File(...),
    lat: float = Form(0.0),
    lng: float = Form(0.0),
    region: str = Form("Unknown Region")
):
    try:
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        image_results = []
        
        # Hyperparameters (Calibrated for RAM / Performance balance)
        CONF_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 30.0))
        MARGIN_THRESHOLD = float(os.getenv("MARGIN_THRESHOLD", 5.0))
        ENTROPY_THRESHOLD = float(os.getenv("ENTROPY_THRESHOLD", 2.0))
        CONSISTENCY_THRESHOLD = float(os.getenv("CONSISTENCY_THRESHOLD", 0.25))
        N_TTA = 5
        TEMPERATURE = 1.1

        for file in files:
            # Load and Pre-process image
            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert("RGB")
            
            # Create Anchor Tensor
            anchor_tensor = transform(image).unsqueeze(0).to(device)
            anchor_tensor.requires_grad_()
            
            # Predict with TTA
            tta_tensors = [tta_transform(image).to(device) for _ in range(N_TTA)]
            batch_tensor = torch.cat([anchor_tensor, torch.stack(tta_tensors)], dim=0)
            
            with torch.enable_grad():
                outputs = model(batch_tensor) / TEMPERATURE
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                
                mean_probs = torch.mean(probabilities, dim=0)
                confidence, predicted_idx = torch.max(mean_probs, 0)
                
                predicted_classes = torch.argmax(probabilities, dim=1)
                consistency_score = (predicted_classes == predicted_idx.item()).sum().item() / (N_TTA + 1)
                
                top3_prob, top3_idx = torch.topk(mean_probs, min(3, len(class_labels)))
                cam = grad_cam.generate(anchor_tensor, predicted_idx.item())

            confidence_val = confidence.item() * 100
            top2_val = top3_prob[1].item() * 100 if len(top3_prob) > 1 else 0.0
            margin = confidence_val - top2_val
            epsilon = 1e-7
            entropy = -torch.sum(mean_probs * torch.log(mean_probs + epsilon)).item()
            
            # Adjudication logic
            is_uncertain = False
            rejection_reason = ""
            if confidence_val < CONF_THRESHOLD:
                is_uncertain = True
                rejection_reason = "Low confidence"
            elif entropy > ENTROPY_THRESHOLD:
                is_uncertain = True
                rejection_reason = "High visual entropy"
            elif margin < MARGIN_THRESHOLD:
                is_uncertain = True
                rejection_reason = "Class ambiguity"
            elif consistency_score < CONSISTENCY_THRESHOLD:
                is_uncertain = True
                rejection_reason = "TTA inconsistency"

            disease_name = "Unknown" if is_uncertain else class_labels[predicted_idx.item()]
            solution_text = f"{rejection_reason}. Please retake." if is_uncertain else solutions.get(disease_name, "No solution found.")
            
            # Heatmap Base64
            original_cv = cv2.cvtColor(np.array(image.resize((224, 224))), cv2.COLOR_RGB2BGR)
            heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
            overlay = np.uint8(heatmap * 0.4 + original_cv * 0.6)
            _, buffer = cv2.imencode('.jpg', overlay)
            cam_base64 = base64.b64encode(buffer).decode('utf-8')

            image_results.append({
                "disease": disease_name,
                "confidence": round(confidence_val, 2),
                "solution": solution_text,
                "cam_base64": f"data:image/jpeg;base64,{cam_base64}",
                "is_uncertain": is_uncertain,
                "consistency_score": round(consistency_score * 100, 2)
            })
            
            # Cleanup to save RAM
            del image, anchor_tensor, tta_tensors, batch_tensor, outputs, probabilities
            gc.collect()

        # Consensus Check
        valid_predictions = [res["disease"] for res in image_results if res["disease"] != "Unknown"]
        status = "stable" if len(valid_predictions) > 0 else "conflict"
        final_disease = max(set(valid_predictions), key=valid_predictions.count) if valid_predictions else "Unknown"
        final_solution = solutions.get(final_disease, "No specific solution available.") if status == "stable" else "Inconclusive result."
        avg_conf = sum([r["confidence"] for r in image_results if r["disease"] == final_disease]) / valid_predictions.count(final_disease) if valid_predictions else 0.0

        if status == "stable":
            conn = get_db()
            conn.execute("INSERT INTO events (disease, confidence, timestamp, lat, lng, region) VALUES (?, ?, ?, ?, ?, ?)",
                        (final_disease, avg_conf, timestamp, lat, lng, region))
            conn.commit()
            conn.close()

        return {
            "final_prediction": final_disease,
            "confidence": round(avg_conf, 2),
            "solution": final_solution,
            "status": status,
            "image_results": image_results,
            "alerts": analyze_outbreaks(region)
        }
        
        local_alerts = analyze_outbreaks(region)
        
        return {
            "final_prediction": final_disease,
            "confidence": round(avg_conf, 2),
            "solution": final_solution,
            "cross_image_consistency": round(cross_consistency * 100, 2),
            "status": status,
            "image_results": image_results,
            "region": region,
            "alerts": local_alerts
        }
    except Exception as e:
        logger.error(f"Inference Engine Crash: {str(e)}")
        return {"error": str(e)}

@app.get("/analytics")
def get_analytics():
    conn = get_db()
    c = conn.cursor()
    
    # Total scans
    c.execute("SELECT COUNT(*) FROM events")
    total = c.fetchone()[0]
    
    # Distribution
    c.execute("SELECT disease, COUNT(*) as cnt FROM events GROUP BY disease")
    distribution = [{"name": r["disease"].replace("___", " ").replace("_", " "), "value": r["cnt"]} for r in c.fetchall()]
    
    # Regional
    c.execute("SELECT region, COUNT(*) as cnt FROM events GROUP BY region")
    regional = [{"region": r["region"], "scans": r["cnt"]} for r in c.fetchall()]
    
    # Trends (Last 7 days)
    trends = []
    for i in range(6, -1, -1):
        date_str = time.strftime("%Y-%m-%d", time.localtime(time.time() - i*86400))
        c.execute("SELECT COUNT(*) FROM events WHERE date(timestamp) = ?", (date_str,))
        trends.append({"date": date_str, "scans": c.fetchone()[0]})
        
    # Global Alerts (Diseases with > 5 scans anywhere in last 24h)
    c.execute("SELECT disease FROM events WHERE timestamp > datetime('now', '-1 day') GROUP BY disease HAVING COUNT(*) > 5")
    global_alerts = [f"Global Outbreak Warning: {r['disease'].replace('___', ' ')}" for r in c.fetchall()]
    
    conn.close()
    return {
        "total_scans": total,
        "distribution": distribution,
        "regional": regional,
        "trends": trends,
        "global_alerts": global_alerts
    }

@app.post("/feedback")
def submit_feedback(disease: str = Form(...), correction: str = Form(...)):
    conn = sqlite3.connect("feedback.db")
    conn.execute("CREATE TABLE IF NOT EXISTS feedback (disease TEXT, correction TEXT)")
    conn.execute("INSERT INTO feedback VALUES (?, ?)", (disease, correction))
    conn.commit()
    conn.close()
    return {"message": "Thank you for your feedback!"}

@app.get("/model-info")
def get_model_info():
    return {
        "version": "2.1.0-organic",
        "architecture": "MobileNetV2",
        "num_classes": len(class_labels),
        "status": "Production-Ready",
        "last_trained": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(MODEL_PATH))) if os.path.exists(MODEL_PATH) else "Never"
    }

class ChatRequest(BaseModel):
    message: str
    disease: str
    language: str



@app.post("/chat")
async def chat(request: ChatRequest):
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        return {"reply": "Please set your GEMINI_API_KEY environment variable to enable the AI Agronomist."}
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-pro')
        prompt = f"You are a helpful agronomist specialized in crop diseases. The user has a plant infected with {request.disease}. Their question is: {request.message}. Respond in {request.language}."
        response = model.generate_content(prompt)
        return {"reply": response.text}
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return {"reply": "I'm having trouble connecting to Google Gemini. Please check your API key and network."}

if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host=host, port=port)
