"""Unit tests for the model architecture, checkpointing, and config loading."""
import sys
from pathlib import Path

import pytest
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from model import SimpleCNN, get_model  # noqa: E402


@pytest.mark.parametrize("architecture", ["resnet18", "simplecnn"])
def test_get_model_returns_expected_class(architecture):
    model = get_model(architecture, num_classes=10)
    if architecture == "resnet18":
        assert type(model).__name__ == "ResNet"
    else:
        assert isinstance(model, SimpleCNN)


@pytest.mark.parametrize("architecture", ["resnet18", "simplecnn"])
def test_forward_pass_output_shape(architecture):
    model = get_model(architecture, num_classes=10)
    model.eval()
    batch = torch.randn(4, 3, 32, 32)
    with torch.no_grad():
        output = model(batch)
    assert output.shape == (4, 10)


def test_get_model_respects_num_classes():
    model = get_model("simplecnn", num_classes=7)
    model.eval()
    batch = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        output = model(batch)
    assert output.shape == (2, 7)


def test_get_model_unknown_architecture_raises():
    with pytest.raises(ValueError):
        get_model("not-a-real-architecture")


def test_checkpoint_save_and_load_roundtrip(tmp_path):
    model = get_model("simplecnn", num_classes=10)
    checkpoint_path = tmp_path / "classifier_v1.pt"

    torch.save(
        {
            "epoch": 1,
            "model_state_dict": model.state_dict(),
            "architecture": "simplecnn",
            "num_classes": 10,
            "val_loss": 1.2345,
            "val_accuracy": 0.42,
        },
        checkpoint_path,
    )

    assert checkpoint_path.exists()

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    restored = get_model(checkpoint["architecture"], num_classes=checkpoint["num_classes"])
    restored.load_state_dict(checkpoint["model_state_dict"])
    restored.eval()

    batch = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        original_output = model.eval()(batch)
        restored_output = restored(batch)
    assert torch.allclose(original_output, restored_output)


def test_training_config_yaml_has_required_keys():
    config_path = Path(__file__).resolve().parent.parent / "configs" / "training_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    assert config["model"]["architecture"] == "resnet18"
    assert config["model"]["num_classes"] == 10
    assert "epochs" in config["training"]
    assert "batch_size" in config["training"]
    assert "learning_rate" in config["training"]
    assert "early_stopping_patience" in config["training"]
    assert config["data"]["dataset"] == "cifar10"
    assert "checkpoint_dir" in config["output"]
    assert "model_name" in config["output"]
