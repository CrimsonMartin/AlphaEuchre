"""
Transformer Neural Network for Euchre Card Playing
Uses self-attention to learn relationships between cards and game state
"""

import torch
import torch.nn as nn
import numpy as np
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""

    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :]


class TransformerEuchreNN(nn.Module):
    """
    Transformer-based neural network for Euchre with separate heads for:
    1. Card playing (main task)
    2. Trump selection (call/pass decisions)
    3. Dealer discard (which card to discard when picking up)

    Uses self-attention to capture relationships between cards and game state.
    """

    def __init__(
        self,
        input_size=130,
        card_output_size=24,
        trump_output_size=5,  # 4 suits + pass
        discard_output_size=24,
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        use_cuda=True,
    ):
        super(TransformerEuchreNN, self).__init__()

        # Architecture identifier
        self.architecture_type = "transformer"

        # Determine device
        self.device = torch.device(
            "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
        )

        self.d_model = d_model

        # Card Playing Head - Transformer based
        # Project input features to d_model dimension
        self.card_input_projection = nn.Linear(input_size, d_model)

        # Positional encoding
        self.card_pos_encoder = PositionalEncoding(d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            batch_first=True,
        )
        self.card_transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # Output head for card playing
        self.card_output = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, card_output_size),
        )

        # Trump Selection Head - smaller transformer
        self.trump_input_projection = nn.Linear(29, d_model // 2)
        self.trump_pos_encoder = PositionalEncoding(d_model // 2)

        trump_encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model // 2,
            nhead=2,
            dim_feedforward=128,
            dropout=0.1,
            batch_first=True,
        )
        self.trump_transformer = nn.TransformerEncoder(trump_encoder_layer, 1)

        self.trump_output = nn.Sequential(
            nn.Linear(d_model // 2, 32),
            nn.ReLU(),
            nn.Linear(32, trump_output_size),
        )

        # Dealer Discard Head
        self.discard_input_projection = nn.Linear(35, d_model // 2)
        self.discard_pos_encoder = PositionalEncoding(d_model // 2)

        discard_encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model // 2,
            nhead=2,
            dim_feedforward=128,
            dropout=0.1,
            batch_first=True,
        )
        self.discard_transformer = nn.TransformerEncoder(discard_encoder_layer, 1)

        self.discard_output = nn.Sequential(
            nn.Linear(d_model // 2, 32),
            nn.ReLU(),
            nn.Linear(32, discard_output_size),
        )

        self.softmax = nn.Softmax(dim=1)

        # Move model to device
        self.to(self.device)

    def forward(self, x):
        """Forward pass through the card playing network"""
        # Project input to d_model dimension
        # x shape: (batch, 130)
        x = self.card_input_projection(x)  # (batch, d_model)

        # Add sequence dimension for transformer
        x = x.unsqueeze(1)  # (batch, 1, d_model)

        # Add positional encoding
        x = self.card_pos_encoder(x)

        # Pass through transformer
        x = self.card_transformer(x)  # (batch, 1, d_model)

        # Take the output (remove sequence dimension)
        x = x.squeeze(1)  # (batch, d_model)

        # Pass through output head
        output = self.card_output(x)

        return self.softmax(output)

    def forward_trump(self, x):
        """Forward pass through the trump selection network"""
        # Project input
        x = self.trump_input_projection(x)  # (batch, d_model//2)

        # Add sequence dimension
        x = x.unsqueeze(1)  # (batch, 1, d_model//2)

        # Add positional encoding
        x = self.trump_pos_encoder(x)

        # Pass through transformer
        x = self.trump_transformer(x)  # (batch, 1, d_model//2)

        # Take the output
        x = x.squeeze(1)  # (batch, d_model//2)

        # Pass through output head
        output = self.trump_output(x)

        return self.softmax(output)

    def forward_discard(self, x):
        """Forward pass through the discard selection network"""
        # Project input
        x = self.discard_input_projection(x)  # (batch, d_model//2)

        # Add sequence dimension
        x = x.unsqueeze(1)  # (batch, 1, d_model//2)

        # Add positional encoding
        x = self.discard_pos_encoder(x)

        # Pass through transformer
        x = self.discard_transformer(x)  # (batch, 1, d_model//2)

        # Take the output
        x = x.squeeze(1)  # (batch, d_model//2)

        # Pass through output head
        output = self.discard_output(x)

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
