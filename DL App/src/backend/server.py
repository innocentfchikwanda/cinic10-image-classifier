import os
import io
from typing import List, Dict, Any

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

# CINIC-10 class names (same as CIFAR-10)
CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

# Define the model architecture matching the training script
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

APP_NAME = "Image Classification API"
CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "feedforward_cinic10.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model at import time
model = FeedforwardNet(num_classes=len(CLASSES)).to(DEVICE)

# Load the model weights
if os.path.exists(CHECKPOINT_PATH):
    try:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        # The checkpoint is a direct state_dict (OrderedDict)
        model.load_state_dict(checkpoint)
        print(f"Successfully loaded model from {CHECKPOINT_PATH}")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Initializing model with random weights")
else:
    print(f"Warning: Checkpoint not found at {CHECKPOINT_PATH}. Using random weights.")

model.eval()

# Preprocessing consistent with the training script
PREPROCESS = T.Compose([
    T.Resize((32, 32)),          # Ensure all images are 32x32
    T.ToTensor(),                # Convert to tensor [0,1]
    T.Normalize(
        mean=[0.4789, 0.4723, 0.4305],    # Training set stats from the script
        std=[0.2421, 0.2383, 0.2587]
    ),
    T.Lambda(lambda x: x.view(-1))  # Flatten (3,32,32) -> (3072,)
])

app = FastAPI(
    title=APP_NAME,
    version="0.1.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)

# CORS for local and static deployments
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the frontend static files - handle both local and cloud paths
def get_frontend_path():
    # Try different possible paths for frontend files
    possible_paths = [
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "src", "frontend", "build"),
        os.path.join(os.path.dirname(__file__), "..", "frontend", "build"),
        os.path.join("/opt/render/project/src", "src", "frontend", "build"),
        os.path.join(os.getcwd(), "src", "frontend", "build"),
        # Fallback paths for old structure
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend_clean", "build"),
        os.path.join(os.getcwd(), "frontend_clean", "build"),
    ]
    
    for path in possible_paths:
        if os.path.exists(path) and os.path.exists(os.path.join(path, "index.html")):
            return path
    
    return None

frontend_path = get_frontend_path()

if frontend_path and os.path.exists(os.path.join(frontend_path, "assets")):
    app.mount("/assets", StaticFiles(directory=os.path.join(frontend_path, "assets")), name="assets")

@app.get("/")
async def root():
    if frontend_path and os.path.exists(os.path.join(frontend_path, "index.html")):
        try:
            return FileResponse(os.path.join(frontend_path, "index.html"))
        except Exception as e:
            return JSONResponse(
                status_code=200,
                content={
                    "message": f"{APP_NAME} is running. Frontend file error: {str(e)}",
                    "frontend_path": frontend_path
                }
            )
    else:
        return JSONResponse(
            status_code=200,
            content={
                "message": f"{APP_NAME} API is running successfully!",
                "status": "Frontend not found - API only mode",
                "api_docs": "/api/docs",
                "health_check": "/api/health",
                "prediction_endpoint": "/api/predict",
                "available_classes": "/api/classes",
                "searched_paths": [
                    os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend_clean", "build"),
                    os.path.join(os.path.dirname(__file__), "..", "frontend_clean", "build"),
                    os.path.join("/opt/render/project/src", "frontend_clean", "build"),
                    os.path.join(os.getcwd(), "frontend_clean", "build"),
                ]
            }
        )

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
    print(f"Starting server on http://localhost:{port}")
    print(f"API documentation available at http://localhost:{port}/api/docs")
    print(f"Frontend should be available at http://localhost:{port}")
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        workers=1
    )
