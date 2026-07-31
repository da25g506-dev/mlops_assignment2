"""CNN model definitions for CIFAR-10 / Fashion-MNIST classification."""
import torch
import torch.nn as nn
from torchvision.models import resnet18

# Some AMD EPYC hosts hit a SIGFPE inside MKL-DNN's convolution kernel
# (oneDNN CPU-dispatch bug); disabling the MKL-DNN backend avoids it at a
# small CPU perf cost and has no effect on correctness or GPU execution.
torch.backends.mkldnn.enabled = False


def _resnet18_for_small_images(num_classes: int, pretrained: bool = False) -> nn.Module:
    """ResNet-18 adapted for 32x32 inputs (CIFAR-10/Fashion-MNIST).

    The stock torchvision ResNet-18 stem (7x7 stride-2 conv + maxpool) was
    designed for 224x224 ImageNet images and downsamples 32x32 inputs to
    nothing useful. We swap in a 3x3 stride-1 stem and drop the maxpool,
    which is the standard adaptation used for CIFAR-scale ResNets.
    """
    weights = "IMAGENET1K_V1" if pretrained else None
    model = resnet18(weights=weights)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


class SimpleCNN(nn.Module):
    """A small from-scratch CNN, offered as a lighter alternative to ResNet-18."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def get_model(architecture: str, num_classes: int = 10, pretrained: bool = False) -> nn.Module:
    """Factory for the model architectures supported by this project."""
    architecture = architecture.lower()
    if architecture == "resnet18":
        return _resnet18_for_small_images(num_classes=num_classes, pretrained=pretrained)
    if architecture == "simplecnn":
        return SimpleCNN(num_classes=num_classes)
    raise ValueError(f"Unknown model architecture: {architecture}")
