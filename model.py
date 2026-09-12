"""A small convolutional network over log-mel segments.

Deliberately small. There are 942 patients, so a large network would memorise
them. Four conv blocks and global pooling keeps the parameter count low enough
that the held-out numbers mean something.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MurmurNet(nn.Module):
    def __init__(self, n_mels: int = 64):
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(block(1, 16), block(16, 32), block(32, 64), block(64, 64))
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(64, 1))

    def forward(self, x):
        h = self.features(x)
        # Mean over time, max over frequency: a murmur is a sustained band of
        # noise, so it should survive averaging along the time axis.
        h = h.mean(dim=3).amax(dim=2)
        return self.head(h).squeeze(1)
