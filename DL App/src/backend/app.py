import os
import json
from typing import Dict, Tuple, Optional

import gradio as gr
import torch
import torchvision.transforms as T
from PIL import Image

from model import build_model, CLASSES, try_load_checkpoint

# Paths and device
CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "feedforward_cinic10.pth")
DEVICE = torch.device("cpu")

# Try to load the checkpoint and obtain a model
model, model_info = try_load_checkpoint(CHECKPOINT_PATH, DEVICE)

# Preprocessing (CIFAR/CINIC-10 stats)
PREPROCESS = T.Compose([
    T.Resize((32, 32)),
    T.ToTensor(),
    T.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)),
])


def predict(image: Image.Image) -> Tuple[Dict[str, float], Dict[str, str]]:
    """
    Returns:
        - Top probabilities dict for gr.Label (e.g., {"cat": 0.83, "dog": 0.1, ...})
        - Info/status dictionary (shown as JSON)
    """
    info = {
        "device": str(DEVICE),
        "model_loaded": model is not None,
        "model_info": model_info,
    }

    if image is None:
        return {}, info

    if model is None:
        info["note"] = (
            "Model is not loaded. Please provide the correct architecture or a full serialized model."
        )
        return {}, info

    img = image.convert("RGB")
    x = PREPROCESS(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits[0], dim=-1).cpu().numpy().tolist()

    # Build mapping of class -> probability
    class_probs = {cls: float(probs[i]) for i, cls in enumerate(CLASSES)}
    # Sort and keep top-3 for display (gr.Label will sort automatically if dict)
    top3 = dict(sorted(class_probs.items(), key=lambda kv: kv[1], reverse=True)[:3])

    return top3, info


def build_interface() -> gr.Blocks:
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # CINIC-10 Image Classifier
        Upload an image to get the predicted class. This demo uses a PyTorch model checkpoint `feedforward_cinic10.pth`.
        If the app can't load the checkpoint because the architecture differs, please provide the original model code.
        """)

        with gr.Row():
            with gr.Column():
                inp = gr.Image(type="pil", label="Input Image")
                btn = gr.Button("Predict", variant="primary")
            with gr.Column():
                label = gr.Label(num_top_classes=3, label="Top Predictions")
                info = gr.JSON(label="Run Info")

        btn.click(fn=predict, inputs=inp, outputs=[label, info])

        # Run once to show model status without needing an image
        demo.load(lambda: ( {}, {"device": str(DEVICE), "model_loaded": model is not None, "model_info": model_info} ), inputs=None, outputs=[label, info])

    return demo


if __name__ == "__main__":
    ui = build_interface()
    # Note: share=True is convenient for testing; you can disable it for local-only
    ui.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)), share=False)
