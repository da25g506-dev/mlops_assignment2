"""Training entrypoint for the CIFAR-10 image classifier.

Reads hyperparameters from a YAML config (path resolved from the
TRAINING_CONFIG_PATH environment variable, falling back to the
conventional mounted-volume location and then a local dev path),
runs a training loop with early stopping, logs metrics as JSON lines
to stdout, and checkpoints the best model seen so far.
"""
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import get_dataloaders  # noqa: E402
from model import get_model  # noqa: E402

DEFAULT_CONFIG_SEARCH_PATH = [
    "/app/configs/training_config.yaml",
    str(Path(__file__).resolve().parent.parent / "configs" / "training_config.yaml"),
]


def resolve_config_path() -> str:
    env_path = os.environ.get("TRAINING_CONFIG_PATH")
    if env_path:
        return env_path
    for candidate in DEFAULT_CONFIG_SEARCH_PATH:
        if Path(candidate).exists():
            return candidate
    raise FileNotFoundError(
        "No training config found. Set TRAINING_CONFIG_PATH or mount one of: "
        f"{DEFAULT_CONFIG_SEARCH_PATH}"
    )


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def main():
    config_path = resolve_config_path()
    config = load_config(config_path)
    print(json.dumps({"event": "config_loaded", "path": config_path}), flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(json.dumps({"event": "device_selected", "device": str(device)}), flush=True)

    model = get_model(
        architecture=config["model"]["architecture"],
        num_classes=config["model"]["num_classes"],
    ).to(device)

    train_loader, val_loader = get_dataloaders(
        data_dir=config["data"]["data_dir"],
        batch_size=config["training"]["batch_size"],
        num_workers=config["training"].get("num_workers", 2),
        subset_fraction=config["data"].get("subset_fraction", 1.0),
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
    )
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    patience_counter = 0
    patience = config["training"]["early_stopping_patience"]

    checkpoint_dir = Path(config["output"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(config["training"]["epochs"]):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        log_entry = {
            "event": "epoch_complete",
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 4),
            "train_accuracy": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_accuracy": round(val_acc, 4),
        }
        print(json.dumps(log_entry), flush=True)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_path = checkpoint_dir / config["output"]["model_name"]
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "architecture": config["model"]["architecture"],
                    "num_classes": config["model"]["num_classes"],
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                },
                save_path,
            )
            print(json.dumps({"event": "checkpoint_saved", "path": str(save_path)}), flush=True)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(json.dumps({"event": "early_stopping", "epoch": epoch + 1}), flush=True)
                break

    print(json.dumps({"event": "training_complete", "best_val_loss": round(best_val_loss, 4)}), flush=True)


if __name__ == "__main__":
    main()
