"""
Basic Neural Network for Euchre Card Playing
"""

import torch
import torch.nn as nn
import numpy as np


class BasicEuchreNN(nn.Module):
    """
    Simple feedforward network for Euchre card selection.

    Input: Game state encoding (65 features)
    Output: Probabilities for each possible card play (24 max)

    Supports CUDA acceleration when available.
    """

    def __init__(self, input_size=65, hidden_size=64, output_size=24, use_cuda=True):
        super(BasicEuchreNN, self).__init__()

        # Determine device (CUDA if available and requested)
        self.device = torch.device(
            "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
        )

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

        # Move model to device
        self.to(self.device)

    def forward(self, x):
        """Forward pass through the network"""
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return self.softmax(x)

    def predict_card(self, game_state_encoding):
        """
        Predict which card to play given a game state.

        Args:
            game_state_encoding: Numpy array or tensor of encoded game state

        Returns:
            Card index with highest probability
        """
        with torch.no_grad():
            if isinstance(game_state_encoding, np.ndarray):
                x = torch.FloatTensor(game_state_encoding).unsqueeze(0).to(self.device)
            else:
                x = game_state_encoding.to(self.device)

            output = self.forward(x)
            return torch.argmax(output, dim=1).cpu().item()

    def predict_cards_batch(self, game_state_encodings):
        """
        Predict cards for multiple game states at once (batch inference).

        This is more efficient on GPU than calling predict_card multiple times.

        Args:
            game_state_encodings: List of numpy arrays or batch tensor

        Returns:
            Numpy array of card indices
        """
        with torch.no_grad():
            if isinstance(game_state_encodings, list):
                batch = torch.stack(
                    [torch.FloatTensor(s) for s in game_state_encodings]
                ).to(self.device)
            else:
                batch = game_state_encodings.to(self.device)

            outputs = self.forward(batch)
            return torch.argmax(outputs, dim=1).cpu().numpy()


def encode_game_state(game_state) -> np.ndarray:
    """
    Encode a game state into a feature vector for neural network input.

    Features include:
    - Player's hand (24 features - one-hot for each card in deck)
    - Trump suit (4 features - one-hot)
    - Cards played in current trick (24 features)
    - Current player position (4 features - one-hot)
    - Team scores (2 features - normalized)
    - Tricks won this hand (2 features)
    - Dealer position (4 features - one-hot)
    - Going alone flag (1 feature)

    Returns:
        Numpy array of size 65
    """
    features = []

    # Card encoding: 9C, 10C, JC, QC, KC, AC, 9D, 10D, ... (24 cards total)
    all_cards = []
    for suit in ["C", "D", "H", "S"]:
        for rank in ["9", "10", "J", "Q", "K", "A"]:
            all_cards.append(f"{rank}{suit}")

    # Player's hand (24 features - one-hot)
    hand_encoding = np.zeros(24)
    if "hand" in game_state:
        for card in game_state["hand"]:
            if card in all_cards:
                hand_encoding[all_cards.index(card)] = 1
    features.extend(hand_encoding)

    # Trump suit (4 features - one-hot)
    trump_encoding = np.zeros(4)
    if game_state.get("trump"):
        trump_map = {"C": 0, "D": 1, "H": 2, "S": 3}
        trump_encoding[trump_map.get(game_state["trump"], 0)] = 1
    features.extend(trump_encoding)

    # Cards played in current trick (24 features - one-hot)
    trick_encoding = np.zeros(24)
    if game_state.get("current_trick") and game_state["current_trick"].get("cards"):
        for card_info in game_state["current_trick"]["cards"]:
            card = card_info.get("card")
            if card in all_cards:
                trick_encoding[all_cards.index(card)] = 1
    features.extend(trick_encoding)

    # Current player position (4 features - one-hot)
    position_encoding = np.zeros(4)
    position = game_state.get("current_player_position", 0)
    position_encoding[position] = 1
    features.extend(position_encoding)

    # Team scores (2 features - normalized to 0-1)
    team1_score = game_state.get("team1_score", 0) / 10.0
    team2_score = game_state.get("team2_score", 0) / 10.0
    features.extend([team1_score, team2_score])

    # Tricks won this hand (2 features)
    team1_tricks = game_state.get("team1_tricks", 0) / 5.0
    team2_tricks = game_state.get("team2_tricks", 0) / 5.0
    features.extend([team1_tricks, team2_tricks])

    # Dealer position (4 features - one-hot)
    dealer_encoding = np.zeros(4)
    dealer_pos = game_state.get("dealer_position", 0)
    dealer_encoding[dealer_pos] = 1
    features.extend(dealer_encoding)

    # Going alone flag (1 feature)
    going_alone = 1.0 if game_state.get("going_alone", False) else 0.0
    features.append(going_alone)

    return np.array(features, dtype=np.float32)
