"""The CNN that classifies a 32x32 grayscale eye crop as closed (0) or open (1).

Three convolutional blocks, each
  conv3x3 -> BatchNorm -> ReLU -> conv3x3 -> BatchNorm -> ReLU -> MaxPool -> Dropout
double the number of channels and halve the image size:

  1x32x32 -> 32x16x16 -> 64x8x8 -> 128x4x4

then global average pooling and a small classifier head give 2 scores (logits).
Small on purpose: it runs on both eyes of every webcam frame in the browser.
"""
import torch
from torch import nn

CLASSES = ["closed", "open"]


def conv_block(in_channels, out_channels, dropout):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        nn.Dropout(dropout),
    )


class EyeCNN(nn.Module):
    def __init__(self, num_classes=2, width=32):
        super().__init__()
        w = width
        self.features = nn.Sequential(
            conv_block(1, w, dropout=0.1),
            conv_block(w, 2 * w, dropout=0.2),
            conv_block(2 * w, 4 * w, dropout=0.3),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(4 * w, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")  # Apple Silicon GPU
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
