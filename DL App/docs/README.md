# CINIC-10 Image Classifier App (Gradio)

This repository contains a simple Gradio web app that performs image classification on the CINIC-10/CIFAR-10 label set using a PyTorch model checkpoint `feedforward_cinic10.pth`.

If the checkpoint architecture differs from the default, the loader will attempt to match it using two candidate models (a small CNN and an MLP). If your checkpoint uses a different architecture, please add your original model definition in `model.py` and adjust `build_model()` accordingly.

## Project Structure

- `app.py`: Gradio UI and inference entrypoint.
- `model.py`: Model definitions and checkpoint loading logic.
- `requirements.txt`: Python dependencies.
- `feedforward_cinic10.pth`: Your trained model checkpoint (provided).

## Local Setup

1. Create and activate a virtual environment

   macOS/Linux:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

   Windows (PowerShell):
   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

2. Install dependencies

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. Run the app

   ```bash
   python app.py
   ```

   Open the local URL printed by Gradio (typically http://127.0.0.1:7860/).

## Customizing the Model

- If your checkpoint is a full serialized model (`torch.save(model, path)`), it will load directly.
- If your checkpoint is a state dict (`torch.save(model.state_dict(), path)` or a dict containing the state dict), the loader tries two defaults: a small CNN and a simple MLP. It selects the architecture with the fewest missing/unexpected keys.
- If you trained a custom architecture, copy the architecture into `model.py`, then modify `build_model()` to return your model. Re-run the app.

## Deployment: Hugging Face Spaces (recommended)

Hugging Face Spaces is the easiest way to deploy this Gradio app.

Option A: Web UI (no CLI)

1. Create a new Space at https://huggingface.co/new-space
2. Choose:
   - Space SDK: Gradio
   - Runtime: CPU
   - Visibility: your choice (Public or Private)
3. After the Space is created, upload the following files via the "Files" tab:
   - `app.py`
   - `model.py`
   - `requirements.txt`
   - `feedforward_cinic10.pth`
   - Optional: `.gitattributes`, `.gitignore`, `README.md`
4. The Space will build automatically and then launch. If build fails due to dependency issues, ensure the `requirements.txt` is present.

Option B: Using `huggingface-cli` (requires token)

1. Install the CLI and login:
   ```bash
   pip install -U "huggingface_hub[cli]"
   huggingface-cli login  # paste your HF token
   ```
2. Create a Space (replace YOUR_USERNAME and MY-CINIC10-APP):
   ```bash
   huggingface-cli repo create YOUR_USERNAME/MY-CINIC10-APP --type=space --space-sdk=gradio
   ```
3. Push your files:
   ```bash
   git init
   git remote add origin https://huggingface.co/spaces/YOUR_USERNAME/MY-CINIC10-APP
   git add .
   git commit -m "Initial commit: CINIC-10 Gradio app"
   git push -u origin main
   ```

> Note: `feedforward_cinic10.pth` is ~15MB which is fine to commit directly. If you prefer Git LFS, ensure `.gitattributes` includes `*.pth filter=lfs diff=lfs merge=lfs -text` and that Git LFS is installed.

## Troubleshooting

- Checkpoint not loading:
  - Inspect the "Run Info" panel in the app for details (e.g., selected arch, missing/unexpected keys).
  - If incompatible, paste your original model architecture in `model.py` and update `build_model()`.
- Slow or failing build on Spaces:
  - Make sure `requirements.txt` is present.
  - Avoid GPU requirements; this app runs on CPU.
- Image looks wrong / poor accuracy:
  - Ensure image normalization matches your training pipeline. Adjust `PREPROCESS` in `app.py` accordingly.

## License

This project is provided for educational purposes. You may adapt it for your coursework or projects.
