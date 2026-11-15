import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)


class CNN(nn.Module):
    """Best/strong baseline from Lab 3: small conv net for Fashion-MNIST."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 10),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


class AttentionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_projection = nn.Linear(49, 64)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64, nhead=4, dim_feedforward=128, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.classifier = nn.Linear(64, 10)

    def forward(self, x):
        # (B,1,28,28) -> 16 patches of 7x7 = 49
        x = x.unfold(2, 7, 7).unfold(3, 7, 7)
        x = x.reshape(x.size(0), 16, 49)
        x = self.patch_projection(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.classifier(x)


def build_model(name: str) -> nn.Module:
    name = name.lower()
    if name == "mlp":
        return MLP()
    if name == "cnn":
        return CNN()
    if name == "attention":
        return AttentionModel()
    raise ValueError(f"Unknown model: {name}")
