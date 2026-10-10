"""eyeTrack1: a small recurrent network that reads 60 seconds of per-frame face features and says
alert / low vigilant / drowsy.

Input: (N, 300, 14): 300 frames (60 s at 5 fps) x 13 normalized features + 1 "face found" flag.
  1. a linear layer mixes the 14 numbers of each frame into 32
  2. a bidirectional GRU reads the 300 frames forwards and backwards (2 x 48 hidden units), so every
     step knows what came before and after: a long closure, a slow blink, a nod
  3. mean and max over time summarize the whole minute ("how much" and "the worst moment")
  4. a small classifier gives 3 scores
"""
import torch
from torch import nn

CLASSES = ["alert", "low vigilant", "drowsy"]


class DrowsinessGRU(nn.Module):
    def __init__(self, n_features=14, hidden=48, n_classes=3, dropout=0.3, proj=32):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(n_features, proj), nn.ReLU(inplace=True), nn.Dropout(0.1))
        self.gru = nn.GRU(proj, hidden, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(4 * hidden, n_classes))

    def forward(self, x):
        h, _ = self.gru(self.inp(x))
        return self.head(torch.cat([h.mean(1), h.amax(1)], 1))
