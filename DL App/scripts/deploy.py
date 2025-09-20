#!/usr/bin/env python3
"""
Deployment script for CINIC-10 Image Classification App
This script handles the deployment setup for cloud platforms
"""
import os
import sys

def main():
    """Main deployment function"""
    print("🚀 Starting CINIC-10 Image Classification App...")
    
    # Change to backend directory
    backend_dir = os.path.join(os.path.dirname(__file__), 'backend')
    if os.path.exists(backend_dir):
        os.chdir(backend_dir)
        print(f"📁 Changed to backend directory: {backend_dir}")
    
    # Import and run the server
    try:
        import server
        print("✅ Server module imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
