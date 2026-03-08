"""
State-Value Critic Network for Actor-Critic RL Training in Euchre
Three separate value heads matching the policy heads in BasicEuchreNN.
"""

import torch
import torch.nn as nn


class EuchreCritic(nn.Module):
    """
    State-value critic with three value heads for Actor-Critic training:
    1. Card value head - estimates V(s) for card playing states
    2. Trump value head - estimates V(s) for trump selection states
    3. Discard value head - estimates V(s) for dealer discard states

    Each head outputs a single scalar value estimate (no softmax).
    Supports CUDA acceleration when available.
    """

    def __init__(self, use_cuda=True):
        super(EuchreCritic, self).__init__()

        # Determine device (CUDA if available and requested)
        self.device = torch.device(
            "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
        )

        # Card value head: 161 -> 64 -> 32 -> 1
        self.card_value_head = nn.Sequential(
            nn.Linear(161, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        # Trump value head: 49 -> 32 -> 16 -> 1
        self.trump_value_head = nn.Sequential(
            nn.Linear(49, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

        # Discard value head: 35 -> 32 -> 1
        self.discard_value_head = nn.Sequential(
            nn.Linear(35, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        # Initialize weights
        self._initialize_weights()

        # Move model to device
        self.to(self.device)

    def _initialize_weights(self):
        """
        Initialize hidden layers with Xavier uniform and output layers
        with small normal distribution (std=0.1).
        """
        for head in [self.card_value_head, self.trump_value_head, self.discard_value_head]:
            layers = [m for m in head.modules() if isinstance(m, nn.Linear)]
            # Hidden layers: Xavier uniform
            for layer in layers[:-1]:
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
            # Output layer: small normal
            output_layer = layers[-1]
            nn.init.normal_(output_layer.weight, mean=0.0, std=0.1)
            nn.init.zeros_(output_layer.bias)

    def forward_card(self, x):
        """
        Forward pass through the card value head.

        Args:
            x: Tensor of card playing state features (batch_size, 161)

        Returns:
            Scalar value estimate V(s) with shape (batch_size, 1)
        """
        return self.card_value_head(x)

    def forward_trump(self, x):
        """
        Forward pass through the trump value head.

        Args:
            x: Tensor of trump selection state features (batch_size, 49)

        Returns:
            Scalar value estimate V(s) with shape (batch_size, 1)
        """
        return self.trump_value_head(x)

    def forward_discard(self, x):
        """
        Forward pass through the discard value head.

        Args:
            x: Tensor of discard state features (batch_size, 35)

        Returns:
            Scalar value estimate V(s) with shape (batch_size, 1)
        """
        return self.discard_value_head(x)
