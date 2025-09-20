#!/usr/bin/env python3
"""
Main server entry point for CINIC-10 Image Classification App
This file imports and runs the actual server from src/backend/
"""
import sys
import os

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import and run the server
if __name__ == "__main__":
    try:
        from backend.server import app
        import uvicorn
        
        port = int(os.environ.get("PORT", 8000))
        print(f"🚀 Starting CINIC-10 Image Classifier on port {port}")
        uvicorn.run(app, host="0.0.0.0", port=port)
    except ImportError:
        # Fallback to minimal server if main server fails
        try:
            from backend.minimal_server import app
            import uvicorn
            
            port = int(os.environ.get("PORT", 8000))
            print(f"🚀 Starting minimal server on port {port}")
            uvicorn.run(app, host="0.0.0.0", port=port)
        except Exception as e:
            print(f"❌ Failed to start server: {e}")
            sys.exit(1)
