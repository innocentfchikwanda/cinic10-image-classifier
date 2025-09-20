#!/usr/bin/env python3
"""
Minimal CINIC-10 Image Classification server for Render deployment
This version prioritizes reliability over features
"""
import os
import sys

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import HTMLResponse
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn
    
    print("✅ FastAPI imports successful")
except ImportError as e:
    print(f"❌ FastAPI import failed: {e}")
    sys.exit(1)

# Simple FastAPI app
app = FastAPI(title="CINIC-10 Image Classifier", version="1.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple HTML page
SIMPLE_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>CINIC-10 Image Classifier</title>
    <style>
        body { 
            font-family: Arial, sans-serif; 
            max-width: 800px; 
            margin: 0 auto; 
            padding: 20px; 
            background: #1a1a1a; 
            color: #fff; 
            text-align: center;
        }
        .success { color: #4CAF50; font-size: 24px; margin: 20px 0; }
        .info { background: #333; padding: 20px; border-radius: 10px; margin: 20px 0; }
        a { color: #4CAF50; text-decoration: none; }
        a:hover { text-decoration: underline; }
    </style>
</head>
<body>
    <h1>🎯 CINIC-10 Image Classifier</h1>
    <div class="success">✅ Deployment Successful!</div>
    
    <div class="info">
        <h3>🚀 Server Status</h3>
        <p>The CINIC-10 image classification server is running successfully on Render!</p>
        <p>This confirms that the deployment infrastructure is working correctly.</p>
    </div>
    
    <div class="info">
        <h3>📋 Available Endpoints</h3>
        <p><strong>Health Check:</strong> <a href="/api/health">/api/health</a></p>
        <p><strong>API Documentation:</strong> <a href="/docs">/docs</a></p>
        <p><strong>Server Info:</strong> <a href="/api/info">/api/info</a></p>
    </div>
    
    <div class="info">
        <h3>🎊 Success!</h3>
        <p>Your CINIC-10 image classification application has been successfully deployed to the cloud!</p>
        <p>The server is running and ready to handle requests.</p>
    </div>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def root():
    return SIMPLE_HTML

@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "message": "CINIC-10 Image Classifier is running",
        "deployment": "render",
        "version": "1.0.0"
    }

@app.get("/api/info")
async def info():
    return {
        "app_name": "CINIC-10 Image Classifier",
        "status": "deployed",
        "platform": "Render",
        "python_version": sys.version,
        "working_directory": os.getcwd(),
        "environment": dict(os.environ),
        "files_in_directory": os.listdir('.') if os.path.exists('.') else []
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"🚀 Starting minimal server on port {port}")
    print(f"🔧 Working directory: {os.getcwd()}")
    print(f"🐍 Python version: {sys.version}")
    
    try:
        uvicorn.run(
            app, 
            host="0.0.0.0", 
            port=port,
            log_level="info",
            access_log=True
        )
    except Exception as e:
        print(f"❌ Server startup failed: {e}")
        import traceback
        traceback.print_exc()
