"""FastAPI inference service for the CIFAR-10 image classifier.

Loads a trained checkpoint on startup and exposes:
  - GET  /health   -> 200 once the model is loaded, 503 otherwise
  - POST /predict   -> multipart/form-data image upload, returns class
                        probabilities

Checkpoint location is resolved from the CHECKPOINT_PATH environment
variable, falling back to the conventional mounted-volume path used by
the Docker/Kubernetes serving setup.
"""
import io
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import get_transforms  # noqa: E402
from model import get_model  # noqa: E402

CIFAR10_CLASSES = [
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

DEFAULT_CHECKPOINT_SEARCH_PATH = [
    "/app/checkpoints/classifier_v1.pt",
    str(Path(__file__).resolve().parent.parent / "checkpoints" / "classifier_v1.pt"),
]

_model: Optional[torch.nn.Module] = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_transform = get_transforms(train=False)


def resolve_checkpoint_path() -> Optional[str]:
    env_path = os.environ.get("CHECKPOINT_PATH")
    if env_path and Path(env_path).exists():
        return env_path
    for candidate in DEFAULT_CHECKPOINT_SEARCH_PATH:
        if Path(candidate).exists():
            return candidate
    return None


def load_model() -> Optional[torch.nn.Module]:
    checkpoint_path = resolve_checkpoint_path()
    if checkpoint_path is None:
        return None
    checkpoint = torch.load(checkpoint_path, map_location=_device, weights_only=False)
    model = get_model(
        architecture=checkpoint.get("architecture", "resnet18"),
        num_classes=checkpoint.get("num_classes", 10),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(_device)
    model.eval()
    return model


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model
    _model = load_model()
    yield


app = FastAPI(title="MLOps PyTorch CIFAR-10 Classifier", lifespan=lifespan)


@app.get("/health")
def health():
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ok"}


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        contents = await image.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}") from exc

    tensor = _transform(img).unsqueeze(0).to(_device)
    with torch.no_grad():
        logits = _model(tensor)
        probabilities = F.softmax(logits, dim=1).squeeze(0).tolist()

    predicted_idx = max(range(len(probabilities)), key=lambda i: probabilities[i])
    return {
        "predicted_class": CIFAR10_CLASSES[predicted_idx],
        "probabilities": {
            CIFAR10_CLASSES[i]: round(p, 6) for i, p in enumerate(probabilities)
        },
    }
