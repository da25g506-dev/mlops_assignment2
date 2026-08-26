"""Tests for the FastAPI serving surface."""
import asyncio
import importlib
import io
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


class DummyClassifier(torch.nn.Module):
    def forward(self, inputs):
        batch_size = inputs.shape[0]
        logits = torch.arange(10, dtype=torch.float32).repeat(batch_size, 1)
        return logits


class FakeUploadFile:
    def __init__(self, contents: bytes):
        self._contents = contents

    async def read(self):
        return self._contents


def test_health_and_predict_with_loaded_model():
    serve = importlib.import_module("serve")
    serve._model = DummyClassifier()

    image = Image.new("RGB", (32, 32), color=(128, 64, 32))
    payload = io.BytesIO()
    image.save(payload, format="PNG")

    assert serve.health() == {"status": "ok"}
    body = asyncio.run(serve.predict(FakeUploadFile(payload.getvalue())))

    assert body["predicted_class"] in serve.CIFAR10_CLASSES
    assert set(body["probabilities"]) == set(serve.CIFAR10_CLASSES)
    assert abs(sum(body["probabilities"].values()) - 1.0) < 1e-4


def test_resolve_checkpoint_path_prefers_existing_env_path(tmp_path, monkeypatch):
    checkpoint_path = tmp_path / "classifier_v1.pt"
    checkpoint_path.write_bytes(b"placeholder")
    monkeypatch.setenv("CHECKPOINT_PATH", str(checkpoint_path))

    serve = importlib.import_module("serve")

    assert serve.resolve_checkpoint_path() == str(checkpoint_path)
