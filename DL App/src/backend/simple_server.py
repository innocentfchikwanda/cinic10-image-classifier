#!/usr/bin/env python3
"""
Simple API-only version of the CINIC-10 Image Classification server
This version focuses on the API functionality and serves a simple HTML page
"""
import os
import io
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from typing import Dict, Any, List
import uvicorn

# Configuration
APP_NAME = "CINIC-10 Image Classifier"
DEVICE = torch.device("cpu")  # Use CPU for cloud deployment
CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "feedforward_cinic10.pth")

# CINIC-10 classes
CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# Model architecture (same as training script)
class FeedforwardNet(nn.Module):
    def __init__(self, num_classes: int = 10):
        super(FeedforwardNet, self).__init__()
        self.fc1 = nn.Linear(3072, 1024)   # first hidden layer
        self.fc2 = nn.Linear(1024, 512)    # second hidden layer
        self.fc3 = nn.Linear(512, 256)     # third hidden layer
        self.fc4 = nn.Linear(256, num_classes)  # output layer

    def forward_net(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)  # logits
        return x

# Image preprocessing
PREPROCESS = T.Compose([
    T.Resize((32, 32)),          # Ensure all images are 32x32
    T.ToTensor(),                # Convert to tensor [0,1]
    T.Normalize(
        mean=[0.4789, 0.4723, 0.4305],    # Training set stats from the script
        std=[0.2421, 0.2383, 0.2587]
    ),
    T.Lambda(lambda x: x.view(-1))  # Flatten (3,32,32) -> (3072,)
])

# Initialize model
model = FeedforwardNet(num_classes=len(CLASSES))
model.to(DEVICE)
model.eval()

# Load the model weights
if os.path.exists(CHECKPOINT_PATH):
    try:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint)
        print(f"✅ Successfully loaded model from {CHECKPOINT_PATH}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("⚠️  Initializing model with random weights")
else:
    print(f"⚠️  Warning: Checkpoint not found at {CHECKPOINT_PATH}. Using random weights.")

# FastAPI app
app = FastAPI(
    title=APP_NAME,
    description="CINIC-10 Image Classification API with PyTorch",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple HTML interface
HTML_INTERFACE = """
<!DOCTYPE html>
<html>
<head>
    <title>CINIC-10 Image Classifier</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; background: #1a1a1a; color: #fff; }
        .container { text-align: center; }
        .upload-area { border: 2px dashed #4CAF50; padding: 40px; margin: 20px 0; border-radius: 10px; background: #2a2a2a; }
        .upload-area:hover { background: #3a3a3a; }
        input[type="file"] { margin: 20px 0; }
        .results { margin: 20px 0; padding: 20px; background: #2a2a2a; border-radius: 10px; }
        .prediction { margin: 10px 0; padding: 10px; background: #4CAF50; border-radius: 5px; }
        .api-info { background: #333; padding: 20px; border-radius: 10px; margin: 20px 0; }
        button { background: #4CAF50; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; }
        button:hover { background: #45a049; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎯 CINIC-10 Image Classifier</h1>
        <p>Upload an image to classify it into one of 10 CINIC-10 categories</p>
        
        <div class="upload-area">
            <h3>📸 Upload Image</h3>
            <input type="file" id="imageInput" accept="image/*">
            <button onclick="classifyImage()">Classify Image</button>
        </div>
        
        <div id="results" class="results" style="display:none;">
            <h3>🔍 Classification Results</h3>
            <div id="predictions"></div>
        </div>
        
        <div class="api-info">
            <h3>🚀 API Endpoints</h3>
            <p><strong>Health Check:</strong> <a href="/api/health">/api/health</a></p>
            <p><strong>API Documentation:</strong> <a href="/api/docs">/api/docs</a></p>
            <p><strong>Available Classes:</strong> <a href="/api/classes">/api/classes</a></p>
            <p><strong>Prediction:</strong> POST /api/predict (with image file)</p>
        </div>
    </div>

    <script>
        async function classifyImage() {
            const fileInput = document.getElementById('imageInput');
            const file = fileInput.files[0];
            
            if (!file) {
                alert('Please select an image file');
                return;
            }
            
            const formData = new FormData();
            formData.append('file', file);
            
            try {
                const response = await fetch('/api/predict', {
                    method: 'POST',
                    body: formData
                });
                
                if (response.ok) {
                    const result = await response.json();
                    displayResults(result);
                } else {
                    alert('Error classifying image');
                }
            } catch (error) {
                alert('Error: ' + error.message);
            }
        }
        
        function displayResults(result) {
            const resultsDiv = document.getElementById('results');
            const predictionsDiv = document.getElementById('predictions');
            
            let html = '';
            result.results.forEach((pred, i) => {
                html += `<div class="prediction">
                    ${i + 1}. ${pred.label}: ${(pred.confidence * 100).toFixed(2)}%
                </div>`;
            });
            
            predictionsDiv.innerHTML = html;
            resultsDiv.style.display = 'block';
        }
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def root():
    return HTML_INTERFACE

@app.get("/api/health")
async def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "device": str(DEVICE),
        "model_loaded": model is not None,
        "num_classes": len(CLASSES),
        "model_architecture": "FeedforwardNet",
        "checkpoint_loaded": os.path.exists(CHECKPOINT_PATH),
        "endpoints": {
            "docs": "/api/docs",
            "predict": "/api/predict",
            "classes": "/api/classes"
        }
    }

@app.get("/api/classes")
async def classes() -> Dict[str, List[str]]:
    return {"classes": CLASSES}

@app.post("/api/predict")
async def predict(file: UploadFile = File(...)) -> Dict[str, Any]:
    import time
    start_time = time.time()
    
    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")

    # Read and preprocess image
    try:
        raw = await file.read()
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read image: {e}")

    # Preprocess the image
    try:
        x = PREPROCESS(img).unsqueeze(0).to(DEVICE)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image preprocessing failed: {e}")

    # Get model predictions
    try:
        inference_start = time.time()
        with torch.no_grad():
            logits = model.forward_net(x)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        inference_time = time.time() - inference_start
            
        # Get top 3 predictions
        top3_indices = probs.argsort()[-3:][::-1]
        
        # Format predictions with class names and confidence scores
        predictions = []
        for idx in top3_indices:
            predictions.append({
                "label": CLASSES[idx],
                "confidence": float(probs[idx]),
                "description": f"This image is classified as {CLASSES[idx]} with {probs[idx]*100:.2f}% confidence."
            })
        
        total_time = time.time() - start_time
        print(f"Prediction completed in {total_time:.3f}s (inference: {inference_time:.3f}s)")
            
        return {
            "success": True,
            "results": predictions,
            "processingTime": round(total_time, 3),
            "inferenceTime": round(inference_time, 3),
            "model": "FeedforwardNet"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"🚀 Starting {APP_NAME} on port {port}")
    print(f"📊 Model loaded: {os.path.exists(CHECKPOINT_PATH)}")
    print(f"🎯 Available classes: {len(CLASSES)}")
    print(f"🔧 Current working directory: {os.getcwd()}")
    print(f"📁 Files in current directory: {os.listdir('.')}")
    
    # Ensure we can bind to the port
    try:
        uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        # Fallback to a simpler HTTP server if uvicorn fails
        import http.server
        import socketserver
        
        class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                self.wfile.write(b"CINIC-10 Classifier - Server Error. Check logs.")
        
        with socketserver.TCPServer(("", port), MyHTTPRequestHandler) as httpd:
            print(f"Fallback server running on port {port}")
            httpd.serve_forever()
