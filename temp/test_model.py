# src/test_model.py

import torch
import torch.nn as nn


class MLPBaseline(nn.Module):
    """
    Simple MLP baseline for tabular grid-level features.

    Input:
        x: [Batch, input_dim]
           Example: [Batch, 2] for no2_mean and nightlight_mean

    Output:
        logits: [Batch, num_classes]
                Example: [Batch, 5] for 5-class hotspot classification
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 64,
        num_classes: int = 5,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        """
        x: [Batch, input_dim]
        """
        logits = self.net(x)
        return logits