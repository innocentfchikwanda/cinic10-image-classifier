import os
from typing import Dict, Tuple, Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

# CINIC-10/CIFAR-10 class names
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


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16x16

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 8x8

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 4x4
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(256 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class SimpleMLP(nn.Module):
    """A simple feedforward MLP for 32x32 RGB images.

    If your checkpoint was trained on a flattened input, this may match better
    than the CNN. Input size: 3*32*32 = 3072
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.fc1 = nn.Linear(3 * 32 * 32, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


def build_model(num_classes: int = len(CLASSES), arch: str = "cnn") -> nn.Module:
    arch = arch.lower()
    if arch == "mlp":
        return SimpleMLP(num_classes=num_classes)
    # default
    return SimpleCNN(num_classes=num_classes)


def _is_state_dict_like(obj: Any) -> bool:
    if not isinstance(obj, dict):
        return False
    if len(obj) == 0:
        return False
    # Heuristic: looks like a typical state_dict mapping of str->Tensor
    return all(isinstance(k, str) and torch.is_tensor(v) for k, v in obj.items())


class DynamicMLP(nn.Module):
    """Builds an MLP whose layers are named fc1, fc2, ... to match a checkpoint.

    dims should be a list like [in_features, h1, h2, ..., out_features].
    """
    def __init__(self, dims: list[int]):
        super().__init__()
        assert len(dims) >= 2, "dims must have at least input and output sizes"
        self.dims = dims
        # Create layers with names fc1..fcN to match common checkpoints
        for i in range(1, len(dims)):
            in_f, out_f = dims[i - 1], dims[i]
            layer = nn.Linear(in_f, out_f)
            setattr(self, f"fc{i}", layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.flatten(x, 1)
        num_layers = len(self.dims) - 1
        for i in range(1, num_layers + 1):
            layer: nn.Linear = getattr(self, f"fc{i}")
            x = layer(x)
            if i != num_layers:
                x = F.relu(x)
        return x


def _build_mlp_from_state_dict(state_dict: Dict[str, Any]) -> Optional[nn.Module]:
    """Attempt to construct a DynamicMLP that matches fc-layer shapes in a checkpoint.

    Looks for keys like fc1.weight, fc2.weight, ... and extracts their shapes.
    Returns a model if at least two consecutive fc layers are found and shapes are valid.
    """
    # Collect (index, weight_shape) for fcN.weight present in state_dict
    fc_shapes: list[tuple[int, torch.Size]] = []
    idx = 1
    while True:
        w_key = f"fc{idx}.weight"
        b_key = f"fc{idx}.bias"
        if w_key in state_dict and b_key in state_dict and torch.is_tensor(state_dict[w_key]):
            fc_shapes.append((idx, state_dict[w_key].shape))
            idx += 1
            continue
        break

    if len(fc_shapes) < 2:
        return None

    # Shapes are [out_features, in_features]
    # Build dims as [in0, out0, out1, ..., outN]
    fc_shapes.sort(key=lambda t: t[0])
    in0 = fc_shapes[0][1][1]
    dims = [int(in0)]
    for _, shape in fc_shapes:
        out_features = int(shape[0])
        dims.append(out_features)

    try:
        model = DynamicMLP(dims)
    except Exception:
        return None
    return model


def try_load_checkpoint(checkpoint_path: str, device: torch.device) -> Tuple[Optional[nn.Module], Dict[str, Any]]:
    info: Dict[str, Any] = {"path": checkpoint_path}

    if not os.path.exists(checkpoint_path):
        info.update({"status": "missing", "message": "Checkpoint file not found."})
        return None, info

    try:
        obj = torch.load(checkpoint_path, map_location=device)
        info["raw_type"] = str(type(obj))
    except Exception as e:
        info.update({"status": "load_failed", "error": repr(e)})
        return None, info

    # Case 1: Full serialized model
    if isinstance(obj, nn.Module):
        model = obj.to(device)
        model.eval()
        info.update({"status": "ok", "format": "full_model"})
        return model, info

    # Case 2: Dictionary (state_dict or checkpoint dict)
    if isinstance(obj, dict):
        state_dict = None
        # Common keys used by various trainers
        for key in ["model_state", "state_dict", "model", "model_state_dict"]:
            if key in obj and _is_state_dict_like(obj[key]):
                state_dict = obj[key]
                info["ckpt_key"] = key
                break
        if state_dict is None:
            # Maybe the dict itself is the state dict
            if _is_state_dict_like(obj):
                state_dict = obj
                info["ckpt_key"] = "<root>"

        # Detect classes metadata if present
        if "classes" in obj and isinstance(obj["classes"], (list, tuple)):
            info["checkpoint_classes"] = list(obj["classes"])
            info["num_checkpoint_classes"] = len(obj["classes"])

        if state_dict is None:
            info.update({
                "status": "no_state_dict",
                "message": "Checkpoint did not contain a recognizable state_dict.",
            })
            return None, info

        # First, try building a dynamic MLP directly from fc-layer shapes in the checkpoint
        dyn_mlp = _build_mlp_from_state_dict(state_dict)
        if dyn_mlp is not None:
            dyn_mlp = dyn_mlp.to(device)
            try:
                missing, unexpected = dyn_mlp.load_state_dict(state_dict, strict=False)
                dyn_mlp.eval()
                info.update({
                    "status": "ok_partial" if (missing or unexpected) else "ok",
                    "format": "state_dict",
                    "strict": False,
                    "selected_arch": "mlp_from_ckpt",
                    "missing_keys": list(missing),
                    "unexpected_keys": list(unexpected),
                })
                return dyn_mlp, info
            except Exception:
                # Fall back to static candidates below
                pass

        # Try multiple candidate architectures and pick the best match
        candidates = [
            ("cnn", lambda: build_model(num_classes=len(CLASSES), arch="cnn")),
            ("mlp", lambda: build_model(num_classes=len(CLASSES), arch="mlp")),
        ]

        best = {
            "score": None,  # lower is better (missing + unexpected)
            "model": None,
            "arch": None,
            "missing": None,
            "unexpected": None,
        }

        for arch_name, ctor in candidates:
            try:
                cand = ctor().to(device)
                missing, unexpected = cand.load_state_dict(state_dict, strict=False)
                score = len(missing) + len(unexpected)
                if best["score"] is None or score < best["score"]:
                    best.update({
                        "score": score,
                        "model": cand,
                        "arch": arch_name,
                        "missing": list(missing),
                        "unexpected": list(unexpected),
                    })
            except Exception:
                # ignore incompatible architectures
                continue

        if best["model"] is None:
            info.update({
                "status": "shape_mismatch",
                "format": "state_dict",
                "message": "State dict incompatible with available default architectures (cnn/mlp). Provide the original model definition.",
            })
            return None, info

        model = best["model"]
        model.eval()
        info.update({
            "status": "ok_partial" if best["score"] else "ok",
            "format": "state_dict",
            "strict": False,
            "selected_arch": best["arch"],
            "missing_keys": best["missing"],
            "unexpected_keys": best["unexpected"],
        })
        return model, info

    # Unknown format
    info.update({"status": "unknown_format", "message": "Unsupported checkpoint format."})
    return None, info
