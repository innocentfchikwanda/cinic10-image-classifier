# 🎯 CINIC-10 Image Classification Web App

A full-stack web application for classifying images using a PyTorch neural network trained on the CINIC-10 dataset.

## 🚀 Quick Start

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Start the server
python server.py

# Visit http://localhost:8000
```

### Cloud Deployment
The app is configured for multiple cloud platforms. See `deployment/` folder for configurations.

## 📁 Project Structure

```
cinic10-image-classifier/
├── 📄 server.py                    # Main server entry point
├── 📄 requirements.txt             # Python dependencies
├── 📄 .gitignore                   # Git ignore rules
├── 
├── 📂 src/                         # Source code
│   ├── 📂 backend/                 # FastAPI backend
│   │   ├── server.py              # Main FastAPI application
│   │   ├── minimal_server.py      # Minimal server for debugging
│   │   ├── simple_server.py       # Simple server with HTML UI
│   │   ├── feedforward_cinic10.pth # Trained PyTorch model (15MB)
│   │   ├── requirements.txt       # Backend dependencies
│   │   └── test_api.py           # API testing script
│   │
│   └── 📂 frontend/               # React frontend
│       ├── build/                # Production build
│       ├── src/                  # React source code
│       ├── package.json          # Node dependencies
│       └── vite.config.ts        # Vite configuration
│
├── 📂 deployment/                 # Deployment configurations
│   ├── Dockerfile                # Docker configuration
│   ├── render.yaml              # Render deployment
│   ├── railway.toml             # Railway deployment
│   └── netlify.toml             # Netlify configuration
│
├── 📂 scripts/                   # Utility scripts
│   ├── deploy.py                # Deployment helper
│   └── start.sh                 # Startup script
│
├── 📂 models/                    # Model training and architecture
│   ├── initial_training_script.py # Original training script
│   └── model.py                 # Model architecture definitions
│
└── 📂 docs/                     # Documentation
    ├── README.md                # This file
    ├── DEPLOYMENT_COMPLETE.md   # Deployment success guide
    ├── CLOUD_DEPLOYMENT_GUIDE.md # Cloud deployment instructions
    ├── DEPLOYMENT_SUCCESS.md    # Technical achievements
    ├── project_log.md          # Development history
    └── full app prompt.txt     # Original requirements
```

## 🎯 Features

- ✅ **Full-stack web application** with React frontend and FastAPI backend
- ✅ **Image classification** for 10 CINIC-10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
- ✅ **Drag & drop upload** with real-time classification
- ✅ **Modern UI/UX** with animations and visual feedback
- ✅ **Fast inference** (<100ms response times)
- ✅ **RESTful API** with comprehensive documentation
- ✅ **Cloud deployment ready** for multiple platforms

## 🔧 Technical Stack

- **Backend**: FastAPI + PyTorch + Python 3.11
- **Frontend**: React + TypeScript + Vite
- **Model**: Custom feedforward neural network
- **Deployment**: Docker, Render, Railway, Netlify ready

## 📊 Performance

- **Inference Time**: ~95ms average
- **Model Size**: 15MB
- **Accuracy**: Trained on CINIC-10 dataset
- **Response Time**: <100ms end-to-end

## 🌐 Live Demo

**Deployed Application**: https://cinic10-image-classifier.onrender.com/

## 🚀 Deployment

Choose your preferred platform:

1. **Render** (Recommended): Use `deployment/render.yaml`
2. **Railway**: Use `deployment/railway.toml`
3. **Docker**: Use `deployment/Dockerfile`
4. **Netlify**: Use `deployment/netlify.toml`

See `docs/CLOUD_DEPLOYMENT_GUIDE.md` for detailed instructions.

## 📝 API Documentation

- **Health Check**: `/api/health`
- **Classification**: `POST /api/predict`
- **Available Classes**: `/api/classes`
- **Interactive Docs**: `/api/docs`

## 🎊 Status

✅ **COMPLETE** - Fully functional and deployed!

---

*Built with ❤️ using React, FastAPI, PyTorch, and modern web technologies*
