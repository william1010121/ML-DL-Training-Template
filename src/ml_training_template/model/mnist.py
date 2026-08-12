"""Small two-convolution MNIST baseline."""

from __future__ import annotations

from torch import Tensor, nn


class MNISTCNN(nn.Module):  # type: ignore[misc]
    """A compact two-layer CNN with resolution-independent pooling."""

    def __init__(
        self,
        *,
        in_channels: int = 1,
        num_classes: int = 10,
        hidden_channels: int = 32,
    ) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(hidden_channels, hidden_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Linear(hidden_channels * 2 * 7 * 7, num_classes)

    def forward(self, inputs: Tensor) -> Tensor:
        features = self.features(inputs)
        return self.classifier(features.flatten(1))


def build_model(*, in_channels: int, num_classes: int, hidden_channels: int) -> MNISTCNN:
    return MNISTCNN(
        in_channels=in_channels,
        num_classes=num_classes,
        hidden_channels=hidden_channels,
    )
