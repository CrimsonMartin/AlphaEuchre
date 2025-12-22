"""
CNN Neural Network for Euchre Card Playing
Uses 1D convolutions to detect card patterns and combinations
"""

import torch
import torch.nn as nn
import numpy as np


class CNNEuchreNN(nn.Module):
    """
    CNN-based neural network for Euchre with separate heads for:
    1. Card playing (main task)
    2. Trump selection (call/pass decisions)
    3. Dealer discard (which card to discard when picking up)

    Uses 1D convolutions to detect patterns in card distributions.
    """

    def __init__(
        self,
        input_size=130,
        card_output_size=24,
        trump_output_size=5,  # 4 suits + pass
        discard_output_size=24,
        use_cuda=True,
    ):
        super(CNNEuchreNN, self).__init__()

        # Architecture identifier
        self.architecture_type = "cnn"

        # Determine device
        self.device = torch.device(
            "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
        )

        # Card Playing Head - CNN based
        # Reshape input: first 24 features are hand, next sections are other card info
        # We'll process card-related features (96 features: hand + trick + previous + position_suit)
        # through CNN and combine with other features through MLP

        # CNN for card pattern detection
        self.card_conv1 = nn.Conv1d(
            in_channels=4, out_channels=32, kernel_size=3, padding=1
        )
        self.card_conv2 = nn.Conv1d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1
        )
        self.card_conv3 = nn.Conv1d(
            in_channels=64, out_channels=32, kernel_size=3, padding=1
        )
        self.card_pool = nn.AdaptiveAvgPool1d(6)  # Output: 32 * 6 = 192 features

        # Batch normalization for stability
        self.card_bn1 = nn.BatchNorm1d(32)
        self.card_bn2 = nn.BatchNorm1d(64)
        self.card_bn3 = nn.BatchNorm1d(32)

        # MLP for non-card features (34 features: trump, position, scores, etc.)
        self.card_mlp_features = nn.Sequential(
            nn.Linear(34, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # Combined network
        # CNN output (192) + MLP features (64) = 256
        self.card_combined = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, card_output_size),
        )

        # Trump Selection Head - smaller CNN
        # Input: 29 features (hand 24 + turned_up 4 + position 1)
        self.trump_conv1 = nn.Conv1d(
            in_channels=1, out_channels=16, kernel_size=3, padding=1
        )
        self.trump_conv2 = nn.Conv1d(
            in_channels=16, out_channels=8, kernel_size=3, padding=1
        )
        self.trump_pool = nn.AdaptiveAvgPool1d(8)  # Output: 8 * 8 = 64 features

        self.trump_fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, trump_output_size),
        )

        # Dealer Discard Head
        # Input: 35 features
        self.discard_conv1 = nn.Conv1d(
            in_channels=1, out_channels=16, kernel_size=3, padding=1
        )
        self.discard_conv2 = nn.Conv1d(
            in_channels=16, out_channels=8, kernel_size=3, padding=1
        )
        self.discard_pool = nn.AdaptiveAvgPool1d(8)  # Output: 8 * 8 = 64 features

        self.discard_fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, discard_output_size),
        )

        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()

        # Move model to device
        self.to(self.device)

    def _reshape_card_features(self, x):
        """
        Reshape card-related features for CNN processing.
        Input x: (batch, 130)

        Card features layout:
        - 0-23: hand (24)
        - 24-27: trump suit (4)
        - 28-51: current trick cards (24)
        - 52-75: previous tricks cards (24)
        - 76-91: position suit encoding (16)
        - 92-95: current position (4)
        - 96-97: team scores (2)
        - 98-99: tricks won (2)
        - 100-103: dealer position (4)
        - 104: going alone (1)
        - 105-108: suit strength (4)
        - 109: trump remaining (1)
        - 110-113: trump caller (4)
        - 114: is calling team (1)
        - 115: trump round (1)
        - 116-119: lead suit (4)
        - 120-123: turned up suit (4)
        - 124: score diff (1)
        - 125: is dealer (1)
        - 126: cards in hand (1)
        - 127: trump in hand (1)
        - 128: partner played (1)
        - 129: lead player (1)

        We'll reshape card-related features (hand, trick, previous, position_suit) into 4 channels of 24 features
        """
        batch_size = x.shape[0]

        # Extract card features (96 features total)
        hand = x[:, 0:24]  # 24 features
        trick = x[:, 28:52]  # 24 features
        previous = x[:, 52:76]  # 24 features
        position_suit = x[:, 76:92]  # 16 features, pad to 24
        position_suit_padded = torch.cat(
            [position_suit, torch.zeros(batch_size, 8, device=x.device)], dim=1
        )

        # Stack into channels: (batch, 4, 24)
        card_features = torch.stack(
            [hand, trick, previous, position_suit_padded], dim=1
        )

        # Extract non-card features (34 features)
        non_card_features = torch.cat(
            [
                x[:, 24:28],  # trump suit (4)
                x[:, 92:130],  # remaining features (38)
            ],
            dim=1,
        )[
            :, :34
        ]  # Take first 34

        return card_features, non_card_features

    def forward(self, x):
        """Forward pass through the card playing network"""
        card_features, non_card_features = self._reshape_card_features(x)

        # CNN path
        conv_out = self.relu(self.card_bn1(self.card_conv1(card_features)))
        conv_out = self.relu(self.card_bn2(self.card_conv2(conv_out)))
        conv_out = self.relu(self.card_bn3(self.card_conv3(conv_out)))
        conv_out = self.card_pool(conv_out)
        conv_out = conv_out.view(conv_out.size(0), -1)  # Flatten: (batch, 192)

        # MLP path for non-card features
        mlp_out = self.card_mlp_features(non_card_features)  # (batch, 64)

        # Combine
        combined = torch.cat([conv_out, mlp_out], dim=1)  # (batch, 256)
        output = self.card_combined(combined)

        return self.softmax(output)

    def forward_trump(self, x):
        """Forward pass through the trump selection network"""
        # Reshape for 1D conv: (batch, 1, 29)
        x = x.unsqueeze(1)

        conv_out = self.relu(self.trump_conv1(x))
        conv_out = self.relu(self.trump_conv2(conv_out))
        conv_out = self.trump_pool(conv_out)
        conv_out = conv_out.view(conv_out.size(0), -1)  # Flatten

        output = self.trump_fc(conv_out)
        return self.softmax(output)

    def forward_discard(self, x):
        """Forward pass through the discard selection network"""
        # Reshape for 1D conv: (batch, 1, 35)
        x = x.unsqueeze(1)

        conv_out = self.relu(self.discard_conv1(x))
        conv_out = self.relu(self.discard_conv2(conv_out))
        conv_out = self.discard_pool(conv_out)
        conv_out = conv_out.view(conv_out.size(0), -1)  # Flatten

        output = self.discard_fc(conv_out)
        return self.softmax(output)

    def predict_card(self, game_state_encoding):
        """Predict which card to play given a game state."""
        with torch.no_grad():
            if isinstance(game_state_encoding, np.ndarray):
                x = torch.FloatTensor(game_state_encoding).unsqueeze(0).to(self.device)
            else:
                x = game_state_encoding.to(self.device)

            output = self.forward(x)
            return torch.argmax(output, dim=1).cpu().item()

    def predict_trump_decision(self, trump_state_encoding):
        """Predict trump decision (which suit to call or pass)."""
        with torch.no_grad():
            if isinstance(trump_state_encoding, np.ndarray):
                x = torch.FloatTensor(trump_state_encoding).unsqueeze(0).to(self.device)
            else:
                x = trump_state_encoding.to(self.device)

            output = self.forward_trump(x)
            return torch.argmax(output, dim=1).cpu().item()

    def predict_discard(self, discard_state_encoding):
        """Predict which card to discard as dealer."""
        with torch.no_grad():
            if isinstance(discard_state_encoding, np.ndarray):
                x = (
                    torch.FloatTensor(discard_state_encoding)
                    .unsqueeze(0)
                    .to(self.device)
                )
            else:
                x = discard_state_encoding.to(self.device)

            output = self.forward_discard(x)
            return torch.argmax(output, dim=1).cpu().item()

    def predict_cards_batch(self, game_state_encodings):
        """Predict cards for multiple game states at once (batch inference)."""
        with torch.no_grad():
            if isinstance(game_state_encodings, list):
                batch = torch.stack(
                    [torch.FloatTensor(s) for s in game_state_encodings]
                ).to(self.device)
            else:
                batch = game_state_encodings.to(self.device)

            outputs = self.forward(batch)
            return torch.argmax(outputs, dim=1).cpu().numpy()
