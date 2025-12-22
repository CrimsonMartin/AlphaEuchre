"""
Enhanced Neural Network for Euchre Card Playing
Multi-head architecture for card playing, trump selection, and dealer discard
"""

import torch
import torch.nn as nn
import numpy as np


class BasicEuchreNN(nn.Module):
    """
    Multi-head neural network for Euchre with separate heads for:
    1. Card playing (main task)
    2. Trump selection (call/pass decisions)
    3. Dealer discard (which card to discard when picking up)

    Supports CUDA acceleration when available.
    """

    def __init__(
        self,
        input_size=130,
        card_hidden_sizes=[256, 128, 64],
        trump_hidden_sizes=[128, 64],
        discard_hidden_sizes=[64],
        card_output_size=24,
        trump_output_size=5,  # 4 suits + pass
        discard_output_size=24,
        use_cuda=True,
    ):
        super(BasicEuchreNN, self).__init__()

        # Determine device (CUDA if available and requested)
        self.device = torch.device(
            "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
        )

        # Card Playing Head (main network)
        card_layers = []
        prev_size = input_size
        for hidden_size in card_hidden_sizes:
            card_layers.append(nn.Linear(prev_size, hidden_size))
            card_layers.append(nn.ReLU())
            card_layers.append(nn.Dropout(0.1))  # Add dropout for regularization
            prev_size = hidden_size
        card_layers.append(nn.Linear(prev_size, card_output_size))
        self.card_network = nn.Sequential(*card_layers)

        # Trump Selection Head (smaller network for trump decisions)
        trump_layers = []
        prev_size = 29  # hand (24) + turned_up_card (4) + position (1)
        for hidden_size in trump_hidden_sizes:
            trump_layers.append(nn.Linear(prev_size, hidden_size))
            trump_layers.append(nn.ReLU())
            prev_size = hidden_size
        trump_layers.append(nn.Linear(prev_size, trump_output_size))
        self.trump_network = nn.Sequential(*trump_layers)

        # Dealer Discard Head (which card to discard)
        discard_layers = []
        prev_size = 35  # hand_with_pickup (25) + trump (4) + position (4) + scores (2)
        for hidden_size in discard_hidden_sizes:
            discard_layers.append(nn.Linear(prev_size, hidden_size))
            discard_layers.append(nn.ReLU())
            prev_size = hidden_size
        discard_layers.append(nn.Linear(prev_size, discard_output_size))
        self.discard_network = nn.Sequential(*discard_layers)

        self.softmax = nn.Softmax(dim=1)

        # Move model to device
        self.to(self.device)

    def forward(self, x):
        """Forward pass through the card playing network"""
        x = self.card_network(x)
        return self.softmax(x)

    def forward_trump(self, x):
        """Forward pass through the trump selection network"""
        x = self.trump_network(x)
        return self.softmax(x)

    def forward_discard(self, x):
        """Forward pass through the discard selection network"""
        x = self.discard_network(x)
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

    def predict_trump_decision(self, trump_state_encoding):
        """
        Predict trump decision (which suit to call or pass).

        Args:
            trump_state_encoding: Numpy array or tensor of encoded trump state

        Returns:
            Decision index (0-3 = suits C,D,H,S, 4 = pass)
        """
        with torch.no_grad():
            if isinstance(trump_state_encoding, np.ndarray):
                x = torch.FloatTensor(trump_state_encoding).unsqueeze(0).to(self.device)
            else:
                x = trump_state_encoding.to(self.device)

            output = self.forward_trump(x)
            return torch.argmax(output, dim=1).cpu().item()

    def predict_discard(self, discard_state_encoding):
        """
        Predict which card to discard as dealer.

        Args:
            discard_state_encoding: Numpy array or tensor of encoded discard state

        Returns:
            Card index to discard
        """
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

    Enhanced encoding with card memory and position tracking.

    Features (130 total):
    - Player's hand (24 features - one-hot for each card in deck)
    - Trump suit (4 features - one-hot)
    - Cards played in current trick (24 features)
    - Cards played in previous tricks this hand (24 features)
    - Card count by position (16 features - 4 positions x 4 suits)
    - Current player position (4 features - one-hot)
    - Team scores (2 features - normalized)
    - Tricks won this hand (2 features)
    - Dealer position (4 features - one-hot)
    - Going alone flag (1 feature)
    - Lead suit strength (4 features - cards remaining in each suit)
    - Trump cards remaining estimate (1 feature)
    - Who called trump (4 features - position one-hot)
    - Is player on calling team (1 feature)
    - Trump called in round 1 or 2 (1 feature)
    - Lead suit of current trick (4 features - one-hot)
    - Turned up card suit (4 features - one-hot)
    - Score differential (1 feature - normalized)
    - Is player dealer (1 feature)
    - Cards remaining in hand (1 feature)
    - Trump cards in own hand (1 feature)
    - Partner already played in trick (1 feature)
    - Lead player position (1 feature)

    Returns:
        Numpy array of size 130
    """
    features = []

    # Card encoding: 9C, 10C, JC, QC, KC, AC, 9D, 10D, ... (24 cards total)
    all_cards = []
    for suit in ["C", "D", "H", "S"]:
        for rank in ["9", "10", "J", "Q", "K", "A"]:
            all_cards.append(f"{rank}{suit}")

    # Player's hand (24 features - one-hot)
    hand_encoding = np.zeros(24)
    hand_cards = []
    if "hand" in game_state:
        for card in game_state["hand"]:
            if card in all_cards:
                hand_encoding[all_cards.index(card)] = 1
                hand_cards.append(card)
    features.extend(hand_encoding)

    # Trump suit (4 features - one-hot)
    trump_encoding = np.zeros(4)
    trump_suit = None
    if game_state.get("trump"):
        trump_map = {"C": 0, "D": 1, "H": 2, "S": 3}
        trump_suit = game_state["trump"]
        trump_encoding[trump_map.get(trump_suit, 0)] = 1
    features.extend(trump_encoding)

    # Cards played in current trick (24 features - one-hot)
    trick_encoding = np.zeros(24)
    current_trick_cards = []
    if game_state.get("current_trick") and game_state["current_trick"].get("cards"):
        for card_info in game_state["current_trick"]["cards"]:
            card = card_info.get("card")
            if card in all_cards:
                trick_encoding[all_cards.index(card)] = 1
                current_trick_cards.append(card)
    features.extend(trick_encoding)

    # Cards played in previous tricks this hand (24 features - one-hot) [NEW]
    previous_tricks_encoding = np.zeros(24)
    if game_state.get("completed_tricks"):
        for trick in game_state["completed_tricks"]:
            if trick.get("cards"):
                for card_info in trick["cards"]:
                    card = card_info.get("card")
                    if card in all_cards:
                        previous_tricks_encoding[all_cards.index(card)] = 1
    features.extend(previous_tricks_encoding)

    # Card count by position (16 features - 4 positions x 4 suits) [NEW]
    # Track which suits each position has played (helps infer what they might have)
    position_suit_encoding = np.zeros(16)
    if game_state.get("completed_tricks"):
        for trick in game_state["completed_tricks"]:
            if trick.get("cards"):
                for card_info in trick["cards"]:
                    card = card_info.get("card")
                    position = card_info.get("position", 0)
                    if card and len(card) >= 2:
                        suit = card[-1]  # Last character is suit
                        suit_map = {"C": 0, "D": 1, "H": 2, "S": 3}
                        if suit in suit_map:
                            idx = position * 4 + suit_map[suit]
                            position_suit_encoding[idx] = min(
                                position_suit_encoding[idx] + 0.2, 1.0
                            )
    features.extend(position_suit_encoding)

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

    # Lead suit strength (4 features - estimate cards remaining in each suit) [NEW]
    suit_strength = np.zeros(4)
    suit_map = {"C": 0, "D": 1, "H": 2, "S": 3}

    # Count cards played in each suit
    played_cards = current_trick_cards + [
        card_info.get("card")
        for trick in game_state.get("completed_tricks", [])
        for card_info in trick.get("cards", [])
        if card_info.get("card")
    ]

    for suit_char, suit_idx in suit_map.items():
        # 6 cards per suit in Euchre
        played_in_suit = sum(
            1 for card in played_cards if card and card[-1] == suit_char
        )
        remaining = (6 - played_in_suit) / 6.0
        suit_strength[suit_idx] = remaining

    features.extend(suit_strength)

    # Trump cards remaining estimate (1 feature) [NEW]
    trump_remaining = 1.0
    if trump_suit:
        trump_played = sum(
            1 for card in played_cards if card and card[-1] == trump_suit
        )
        trump_remaining = (6 - trump_played) / 6.0
    features.append(trump_remaining)

    # Who called trump - position (4 features - one-hot) [NEW]
    trump_caller_encoding = np.zeros(4)
    trump_caller = game_state.get("trump_caller_position")
    if trump_caller is not None:
        trump_caller_encoding[trump_caller] = 1
    features.extend(trump_caller_encoding)

    # Is player on calling team (1 feature) [NEW]
    is_calling_team = 0.0
    if trump_caller is not None:
        # Team 1: positions 0,2  Team 2: positions 1,3
        player_team = position % 2
        caller_team = trump_caller % 2
        is_calling_team = 1.0 if player_team == caller_team else 0.0
    features.append(is_calling_team)

    # Trump called in round 1 or 2 (1 feature) [NEW]
    trump_round = (
        game_state.get("trump_round", 0) / 2.0
    )  # Normalize: 0=unknown, 0.5=round1, 1.0=round2
    features.append(trump_round)

    # Lead suit of current trick (4 features - one-hot) [NEW]
    lead_suit_encoding = np.zeros(4)
    if game_state.get("current_trick") and game_state["current_trick"].get("cards"):
        cards_in_trick = game_state["current_trick"]["cards"]
        if cards_in_trick:
            lead_card = cards_in_trick[0].get("card")
            if lead_card and len(lead_card) >= 2:
                lead_suit = lead_card[-1]
                if lead_suit in suit_map:
                    lead_suit_encoding[suit_map[lead_suit]] = 1
    features.extend(lead_suit_encoding)

    # Turned up card suit (4 features - one-hot) [NEW]
    turned_up_encoding = np.zeros(4)
    turned_up = game_state.get("turned_up_card")
    if turned_up and len(turned_up) >= 2:
        turned_up_suit = turned_up[-1]
        if turned_up_suit in suit_map:
            turned_up_encoding[suit_map[turned_up_suit]] = 1
    features.extend(turned_up_encoding)

    # Score differential (1 feature - normalized) [NEW]
    score_diff = (
        game_state.get("team1_score", 0) - game_state.get("team2_score", 0)
    ) / 10.0
    # Flip sign if player is on team 2
    if position % 2 == 1:
        score_diff = -score_diff
    features.append(score_diff)

    # Is player dealer (1 feature) [NEW]
    is_dealer = 1.0 if position == dealer_pos else 0.0
    features.append(is_dealer)

    # Cards remaining in hand (1 feature - normalized) [NEW]
    cards_in_hand = len(game_state.get("hand", [])) / 5.0
    features.append(cards_in_hand)

    # Trump cards in own hand (1 feature - normalized) [NEW]
    trump_in_hand = 0.0
    if trump_suit and "hand" in game_state:
        trump_count = sum(
            1 for card in game_state["hand"] if card and card[-1] == trump_suit
        )
        trump_in_hand = trump_count / 5.0
    features.append(trump_in_hand)

    # Partner already played in trick (1 feature) [NEW]
    partner_played = 0.0
    if game_state.get("current_trick") and game_state["current_trick"].get("cards"):
        partner_position = (position + 2) % 4
        for card_info in game_state["current_trick"]["cards"]:
            if card_info.get("position") == partner_position:
                partner_played = 1.0
                break
    features.append(partner_played)

    # Lead player position (1 feature - normalized) [NEW]
    lead_player = 0.0
    if game_state.get("current_trick") and game_state["current_trick"].get("cards"):
        cards_in_trick = game_state["current_trick"]["cards"]
        if cards_in_trick:
            lead_player = cards_in_trick[0].get("position", 0) / 3.0
    features.append(lead_player)

    return np.array(features, dtype=np.float32)


def encode_trump_state(game_state, turned_up_card=None) -> np.ndarray:
    """
    Encode state for trump selection decision.

    Features (29 total):
    - Player's hand (24 features)
    - Turned up card suit (4 features - one-hot)
    - Player position relative to dealer (1 feature - normalized)

    Returns:
        Numpy array of size 29
    """
    features = []

    # Card encoding
    all_cards = []
    for suit in ["C", "D", "H", "S"]:
        for rank in ["9", "10", "J", "Q", "K", "A"]:
            all_cards.append(f"{rank}{suit}")

    # Player's hand (24 features)
    hand_encoding = np.zeros(24)
    if "hand" in game_state:
        for card in game_state["hand"]:
            if card in all_cards:
                hand_encoding[all_cards.index(card)] = 1
    features.extend(hand_encoding)

    # Turned up card suit (4 features - one-hot)
    turned_up_encoding = np.zeros(4)
    if turned_up_card and len(turned_up_card) >= 2:
        suit_map = {"C": 0, "D": 1, "H": 2, "S": 3}
        suit = turned_up_card[-1]
        if suit in suit_map:
            turned_up_encoding[suit_map[suit]] = 1
    features.extend(turned_up_encoding)

    # Position relative to dealer (1 feature - normalized)
    current_pos = game_state.get("current_player_position", 0)
    dealer_pos = game_state.get("dealer_position", 0)
    relative_pos = (current_pos - dealer_pos) % 4
    features.append(relative_pos / 3.0)  # Normalize to 0-1

    return np.array(features, dtype=np.float32)


def encode_discard_state(game_state, hand_with_pickup) -> np.ndarray:
    """
    Encode state for dealer discard decision.

    Features (35 total):
    - Hand with picked up card (25 features - one-hot, 24 cards + 1 for pickup)
    - Trump suit (4 features - one-hot)
    - Dealer position (4 features - one-hot)
    - Team scores (2 features)

    Returns:
        Numpy array of size 35
    """
    features = []

    # Card encoding
    all_cards = []
    for suit in ["C", "D", "H", "S"]:
        for rank in ["9", "10", "J", "Q", "K", "A"]:
            all_cards.append(f"{rank}{suit}")

    # Hand with pickup (25 features - 24 regular + 1 overflow)
    hand_encoding = np.zeros(25)
    for i, card in enumerate(hand_with_pickup[:25]):
        if card in all_cards:
            hand_encoding[all_cards.index(card)] = 1
    features.extend(hand_encoding)

    # Trump suit (4 features)
    trump_encoding = np.zeros(4)
    if game_state.get("trump"):
        trump_map = {"C": 0, "D": 1, "H": 2, "S": 3}
        trump_encoding[trump_map.get(game_state["trump"], 0)] = 1
    features.extend(trump_encoding)

    # Dealer position (4 features)
    dealer_encoding = np.zeros(4)
    dealer_pos = game_state.get("dealer_position", 0)
    dealer_encoding[dealer_pos] = 1
    features.extend(dealer_encoding)

    # Team scores (2 features)
    team1_score = game_state.get("team1_score", 0) / 10.0
    team2_score = game_state.get("team2_score", 0) / 10.0
    features.extend([team1_score, team2_score])

    return np.array(features, dtype=np.float32)
