"""
Policy Gradient Trainer for Euchre AI
Uses REINFORCE algorithm with reward shaping for active play
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import copy
import sys
import os
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "shared"))

from euchre_core.game import EuchreGame, GamePhase
from euchre_core.player import PlayerType
from euchre_core.card import Card, Suit
from networks.basic_nn import (
    BasicEuchreNN,
    encode_game_state,
    encode_trump_state,
    encode_discard_state,
)


class Episode:
    """Records a single game episode with all decisions and rewards"""

    def __init__(self):
        # Store decisions by type: (state, action_idx, log_prob, entropy, position)
        self.card_decisions = []
        self.trump_decisions = []
        self.discard_decisions = []

        # Per-decision rewards: each decision gets its own immediate reward
        # so we can compute proper temporal credit assignment
        self.card_rewards = []  # One reward per card decision
        self.trump_rewards = []  # One reward per trump decision
        self.discard_rewards = []  # One reward per discard decision

        # Game-level reward applied to all decisions as a final bonus
        self.game_reward = 0.0

        # Track game statistics
        self.team1_score = 0
        self.team2_score = 0
        self.tricks_won = defaultdict(int)
        self.trump_calls = defaultdict(int)
        self.euchres = defaultdict(int)


class PolicyGradientTrainer:
    """Trains a single model using policy gradient (REINFORCE) with optional
    self-play, critic baseline, and reward shaping."""

    def __init__(
        self,
        model: BasicEuchreNN,
        learning_rate: float = 0.0003,
        gamma: float = 0.99,
        entropy_beta: float = 0.01,
        exploration_rate: float = 0.1,
        use_cuda: bool = True,
        self_play: bool = False,
        opponent_update_interval: int = 20,
        critic=None,
        critic_lr: float = 0.001,
    ):
        self.model = model
        self.gamma = gamma
        self.entropy_beta = entropy_beta
        self.exploration_rate = exploration_rate

        # Determine device
        self.device = torch.device(
            "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # Critic (optional - for Actor-Critic variance reduction)
        self.critic = critic
        self.critic_optimizer = None
        if self.critic is not None:
            self.critic.to(self.device)
            self.critic_optimizer = optim.Adam(
                self.critic.parameters(), lr=critic_lr
            )

        # Self-play: frozen opponent model + pool of past snapshots
        self.self_play = self_play
        self.opponent_update_interval = opponent_update_interval
        self.opponent_model = None
        self.opponent_pool = []  # List of state_dicts
        self.updates_since_opponent_refresh = 0
        if self.self_play:
            self.update_opponent()

        # Card mapping
        self.all_cards = []
        for suit in ["C", "D", "H", "S"]:
            for rank in ["9", "10", "J", "Q", "K", "A"]:
                self.all_cards.append(f"{rank}{suit}")

        # Statistics
        self.total_games = 0
        self.total_wins = 0
        self.avg_reward = 0.0
        self.running_reward = None

    def update_opponent(self):
        """Deep copy current model as the frozen opponent for self-play."""
        self.opponent_model = copy.deepcopy(self.model)
        self.opponent_model.eval()
        for param in self.opponent_model.parameters():
            param.requires_grad = False
        # Keep pool of past snapshots (max 10)
        self.opponent_pool.append(copy.deepcopy(self.model.state_dict()))
        if len(self.opponent_pool) > 10:
            self.opponent_pool.pop(0)
        self.updates_since_opponent_refresh = 0

    def _select_opponent_for_game(self):
        """Pick an opponent: 70% latest, 30% random from pool."""
        if not self.opponent_pool:
            return self.opponent_model
        if random.random() < 0.7 or len(self.opponent_pool) <= 1:
            return self.opponent_model
        # Load a random past snapshot
        snapshot = random.choice(self.opponent_pool[:-1])
        opp = copy.deepcopy(self.model)
        opp.load_state_dict(snapshot)
        opp.eval()
        for param in opp.parameters():
            param.requires_grad = False
        return opp

    def _opponent_select_action(
        self, opponent, state_encoding, valid_indices, decision_type
    ):
        """Select action using frozen opponent model (no gradient)."""
        if opponent is None:
            return random.choice(valid_indices)

        with torch.no_grad():
            state_tensor = (
                torch.FloatTensor(state_encoding).unsqueeze(0).to(self.device)
            )
            if decision_type == "card":
                logits = opponent.forward(state_tensor, return_logits=True)
            elif decision_type == "trump":
                logits = opponent.forward_trump(state_tensor, return_logits=True)
            else:
                logits = opponent.forward_discard(state_tensor, return_logits=True)

            mask = torch.full_like(logits, float("-inf"))
            mask[0, valid_indices] = 0
            masked_logits = logits + mask
            dist = torch.distributions.Categorical(logits=masked_logits)
            return dist.sample().item()

    @staticmethod
    def compute_hand_strength(hand_cards: List[str], trump_suit_char: str) -> float:
        """Compute a hand strength score for reward shaping when calling trump.
        Returns a value in [0, 1] representing how strong the hand is."""
        if not trump_suit_char or not hand_cards:
            return 0.0

        same_color_map = {"C": "S", "S": "C", "D": "H", "H": "D"}
        left_bower_suit = same_color_map.get(trump_suit_char)

        trump_count = sum(
            1 for c in hand_cards
            if c and len(c) >= 2 and c[-1] == trump_suit_char
        )
        has_right_bower = 1.0 if f"J{trump_suit_char}" in hand_cards else 0.0
        has_left_bower = (
            1.0
            if left_bower_suit and f"J{left_bower_suit}" in hand_cards
            else 0.0
        )
        off_aces = sum(
            1 for c in hand_cards
            if c and len(c) >= 2 and c[0] == "A" and c[-1] != trump_suit_char
        )

        # Weighted strength: bowers are critical, trump count matters, off-aces help
        strength = (
            trump_count / 5.0 * 2.0
            + has_right_bower * 3.0
            + has_left_bower * 2.5
            + off_aces / 3.0 * 1.5
        ) / 10.0
        return min(strength, 1.0)

    def select_action_with_exploration(
        self, state_encoding: np.ndarray, valid_indices: List[int], decision_type: str
    ) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        Select action using model's policy with epsilon-greedy exploration.
        Uses logits from the network (not softmax output) to avoid double-softmax.

        Returns:
            (action_index, log_probability, entropy)
        """
        state_tensor = torch.FloatTensor(state_encoding).unsqueeze(0).to(self.device)

        # Get logits (not probabilities) from network
        if decision_type == "card":
            logits = self.model.forward(state_tensor, return_logits=True)
        elif decision_type == "trump":
            logits = self.model.forward_trump(state_tensor, return_logits=True)
        else:  # discard
            logits = self.model.forward_discard(state_tensor, return_logits=True)

        # Mask invalid actions by setting logits to -inf
        mask = torch.full_like(logits, float("-inf"))
        mask[0, valid_indices] = 0
        masked_logits = logits + mask

        # Single softmax on logits to get proper distribution
        dist = torch.distributions.Categorical(logits=masked_logits)
        entropy = dist.entropy()

        # Epsilon-greedy: sometimes pick random action, but always use
        # the network's log_prob for gradient (keeps gradient flow intact)
        if random.random() < self.exploration_rate:
            action_idx = random.choice(valid_indices)
            action_tensor = torch.tensor(action_idx, device=self.device)
        else:
            action_tensor = dist.sample()
            action_idx = action_tensor.item()

        log_prob = dist.log_prob(action_tensor)

        return action_idx, log_prob, entropy

    def play_game(self) -> Episode:
        """
        Play a single game with the model controlling all 4 positions (self-play).
        Each position learns from its own perspective.

        Per-decision rewards are tracked so each decision gets credit for
        outcomes it directly influenced (not a flat global return).
        """
        episode = Episode()

        # Select opponent for this game
        opponent = (
            self._select_opponent_for_game() if self.self_play else None
        )

        # Create game
        game = EuchreGame(f"training-{random.randint(1000, 9999)}")

        # Add players
        for i in range(4):
            game.add_player(f"Player{i}", PlayerType.RANDOM_AI)

        # Track hand-level info
        current_hand_info = {
            "caller_position": None,
            "calling_team": None,
            "calling_trump_suit_char": None,
            "calling_hand_cards": None,
            "tricks_won_by_pos": {0: 0, 1: 0, 2: 0, 3: 0},
            "was_forced_call": False,
        }

        # Track which decisions belong to the current hand so we can
        # assign hand-outcome rewards to only those decisions
        hand_card_start_idx = 0
        hand_trump_start_idx = 0
        hand_discard_start_idx = 0

        # Start game
        game.start_new_hand()

        # Play until game over
        while game.state.phase != GamePhase.GAME_OVER:
            current_pos = game.state.current_player_position

            # Handle different game phases
            if game.state.phase == GamePhase.TRUMP_SELECTION_ROUND1:
                game_state_dict = game.get_state(perspective_position=current_pos)
                turned_up_card = (
                    str(game.state.turned_up_card)
                    if game.state.turned_up_card
                    else None
                )
                trump_state = encode_trump_state(game_state_dict, turned_up_card)

                suit_map_enum = {
                    Suit.CLUBS: 0,
                    Suit.DIAMONDS: 1,
                    Suit.HEARTS: 2,
                    Suit.SPADES: 3,
                }
                turned_up_suit_idx = None
                if game.state.turned_up_card:
                    turned_up_suit_idx = suit_map_enum.get(
                        game.state.turned_up_card.suit
                    )

                if turned_up_suit_idx is not None:
                    valid_trump_indices = [turned_up_suit_idx, 4]
                else:
                    valid_trump_indices = [4]

                # Model makes decision for all positions
                decision_idx, log_prob = self.select_action_with_exploration(
                    trump_state, valid_trump_indices, "trump"
                )
                episode.trump_decisions.append(
                    (trump_state, decision_idx, log_prob, current_pos)
                )

                # Execute decision
                if decision_idx == 4:
                    try:
                        result = game.pass_trump()
                        # Check if everyone passed (hand_over)
                        if result == "hand_over":
                            # Apply penalty for not calling trump to ALL positions
                            hand_reward = -0.05
                            for pos in range(4):
                                episode.hand_rewards[pos].append(hand_reward)
                            # Start new hand
                            if game.state.phase != GamePhase.GAME_OVER:
                                game.start_new_hand()
                            continue
                    except:
                        if game.state.turned_up_card:
                            game.call_trump(game.state.turned_up_card.suit)
                            current_hand_info["caller_position"] = current_pos
                            current_hand_info["calling_team"] = (
                                1 if current_pos in [0, 2] else 2
                            )
                            episode.trump_calls[current_pos] += 1
                else:
                    try:
                        if game.state.turned_up_card:
                            game.call_trump(game.state.turned_up_card.suit)
                            current_hand_info["caller_position"] = current_pos
                            current_hand_info["calling_team"] = (
                                1 if current_pos in [0, 2] else 2
                            )
                            episode.trump_calls[current_pos] += 1
                    except:
                        try:
                            result = game.pass_trump()
                            if result == "hand_over":
                                hand_reward = -0.05
                                for pos in range(4):
                                    episode.hand_rewards[pos].append(hand_reward)
                                if game.state.phase != GamePhase.GAME_OVER:
                                    game.start_new_hand()
                                continue
                        except:
                            if game.state.turned_up_card:
                                game.call_trump(game.state.turned_up_card.suit)
                                current_hand_info["caller_position"] = current_pos
                                current_hand_info["calling_team"] = (
                                    1 if current_pos in [0, 2] else 2
                                )
                                episode.trump_calls[current_pos] += 1

            elif game.state.phase == GamePhase.TRUMP_SELECTION_ROUND2:
                game_state_dict = game.get_state(perspective_position=current_pos)
                turned_up_card = (
                    str(game.state.turned_up_card)
                    if game.state.turned_up_card
                    else None
                )
                trump_state = encode_trump_state(game_state_dict, turned_up_card)

                turned_up_suit = (
                    game.state.turned_up_card.suit
                    if game.state.turned_up_card
                    else Suit.CLUBS
                )
                is_dealer = (
                    game.state.current_player_position == game.state.dealer_position
                )

                suit_map_enum = {
                    Suit.CLUBS: 0,
                    Suit.DIAMONDS: 1,
                    Suit.HEARTS: 2,
                    Suit.SPADES: 3,
                }
                turned_up_suit_idx = suit_map_enum.get(turned_up_suit, 0)

                valid_trump_indices = [i for i in range(4) if i != turned_up_suit_idx]
                if not is_dealer:
                    valid_trump_indices.append(4)

                # Model makes decision for all positions
                decision_idx, log_prob = self.select_action_with_exploration(
                    trump_state, valid_trump_indices, "trump"
                )
                episode.trump_decisions.append(
                    (trump_state, decision_idx, log_prob, current_pos)
                )

                # Execute decision
                if decision_idx == 4:
                    # Pass trump
                    try:
                        result = game.pass_trump()
                        # Check if everyone passed (hand_over)
                        if result == "hand_over":
                            # Apply penalty for not calling trump to ALL positions
                            hand_reward = -0.05
                            for pos in range(4):
                                episode.hand_rewards[pos].append(hand_reward)
                            # Reset hand info
                            current_hand_info = {
                                "caller_position": None,
                                "calling_team": None,
                                "tricks_won_by_pos": {0: 0, 1: 0, 2: 0, 3: 0},
                            }
                            # Start new hand
                            if game.state.phase != GamePhase.GAME_OVER:
                                game.start_new_hand()
                            continue
                    except:
                        # Fallback if pass fails
                        available_suits = [
                            s
                            for s in [
                                Suit.CLUBS,
                                Suit.DIAMONDS,
                                Suit.HEARTS,
                                Suit.SPADES,
                            ]
                            if s != turned_up_suit
                        ]
                        game.call_trump(random.choice(available_suits))
                        current_hand_info["caller_position"] = current_pos
                        current_hand_info["calling_team"] = (
                            1 if current_pos in [0, 2] else 2
                        )
                        episode.trump_calls[current_pos] += 1
                else:
                    suit_map = [Suit.CLUBS, Suit.DIAMONDS, Suit.HEARTS, Suit.SPADES]
                    selected_suit = suit_map[decision_idx]

                    if selected_suit != turned_up_suit:
                        game.call_trump(selected_suit)
                        current_hand_info["caller_position"] = current_pos
                        current_hand_info["calling_team"] = (
                            1 if current_pos in [0, 2] else 2
                        )
                        episode.trump_calls[current_pos] += 1
                    else:
                        available_suits = [
                            s
                            for s in [
                                Suit.CLUBS,
                                Suit.DIAMONDS,
                                Suit.HEARTS,
                                Suit.SPADES,
                            ]
                            if s != turned_up_suit
                        ]
                        if is_dealer:
                            game.call_trump(random.choice(available_suits))
                            current_hand_info["caller_position"] = current_pos
                            current_hand_info["calling_team"] = (
                                1 if current_pos in [0, 2] else 2
                            )
                            episode.trump_calls[current_pos] += 1
                        else:
                            try:
                                result = game.pass_trump()
                                if result == "hand_over":
                                    hand_reward = -0.05
                                    for pos in range(4):
                                        episode.hand_rewards[pos].append(hand_reward)
                                    current_hand_info = {
                                        "caller_position": None,
                                        "calling_team": None,
                                        "tricks_won_by_pos": {0: 0, 1: 0, 2: 0, 3: 0},
                                    }
                                    if game.state.phase != GamePhase.GAME_OVER:
                                        game.start_new_hand()
                                    continue
                            except:
                                game.call_trump(random.choice(available_suits))
                                current_hand_info["caller_position"] = current_pos
                                current_hand_info["calling_team"] = (
                                    1 if current_pos in [0, 2] else 2
                                )
                                episode.trump_calls[current_pos] += 1

            elif game.state.phase == GamePhase.DEALER_DISCARD:
                dealer = game.state.get_player(game.state.dealer_position)
                game_state_dict = game.get_state(
                    perspective_position=game.state.dealer_position
                )
                hand_with_pickup = [str(card) for card in dealer.hand]
                discard_state = encode_discard_state(game_state_dict, hand_with_pickup)

                valid_discard_indices = []
                for card in dealer.hand:
                    card_str = str(card)
                    if card_str in self.all_cards:
                        valid_discard_indices.append(self.all_cards.index(card_str))

                # Model makes decision for all positions (including dealer)
                if valid_discard_indices:
                    card_idx, log_prob = self.select_action_with_exploration(
                        discard_state, valid_discard_indices, "discard"
                    )
                    episode.discard_decisions.append(
                        (discard_state, card_idx, log_prob, game.state.dealer_position)
                    )
                else:
                    card_idx = 0

                predicted_card_str = (
                    self.all_cards[card_idx] if card_idx < len(self.all_cards) else None
                )

                card_to_discard = None
                if predicted_card_str:
                    for card in dealer.hand:
                        if str(card) == predicted_card_str:
                            card_to_discard = card
                            break

                if card_to_discard is None:
                    card_to_discard = dealer.hand[0]

                game.dealer_discard(card_to_discard)

            elif game.state.phase == GamePhase.PLAYING:
                game_state_dict = game.get_state(perspective_position=current_pos)
                state_encoding = encode_game_state(game_state_dict)
                valid_cards = game.get_valid_moves(current_pos)

                if valid_cards:
                    valid_card_indices = []
                    for card in valid_cards:
                        card_str = str(card)
                        if card_str in self.all_cards:
                            valid_card_indices.append(self.all_cards.index(card_str))

                    # Model makes decision for all positions
                    if valid_card_indices:
                        card_idx, log_prob = self.select_action_with_exploration(
                            state_encoding, valid_card_indices, "card"
                        )
                        episode.card_decisions.append(
                            (state_encoding, card_idx, log_prob, current_pos)
                        )
                    else:
                        card_idx = 0

                    predicted_card_str = (
                        self.all_cards[card_idx]
                        if card_idx < len(self.all_cards)
                        else None
                    )

                    selected_card = None
                    if predicted_card_str:
                        for card in valid_cards:
                            if str(card) == predicted_card_str:
                                selected_card = card
                                break

                    if selected_card is None:
                        selected_card = valid_cards[0]

                    result = game.play_card(selected_card)

                    # TRICK-LEVEL REWARDS - Give to each position from their perspective
                    if result.get("trick_complete"):
                        winner_pos = result["trick_winner"]
                        current_hand_info["tricks_won_by_pos"][winner_pos] += 1

                        # Reward for winning trick (from each team's perspective)
                        for pos in range(4):
                            if (pos % 2) == (winner_pos % 2):  # Same team
                                trick_reward = 0.02
                                # Bonus if this team called trump
                                if current_hand_info["calling_team"] == (
                                    1 if pos % 2 == 0 else 2
                                ):
                                    trick_reward += 0.01
                                episode.trick_rewards[pos].append(trick_reward)

                    # Check if hand complete
                    if result.get("hand_complete"):
                        # HAND-LEVEL REWARDS - Give to each position from their perspective
                        hand_winner = result["hand_winner"]
                        winning_team = hand_winner["winning_team"]
                        points_awarded = hand_winner["points_awarded"]
                        calling_team = current_hand_info["calling_team"]

                        for pos in range(4):
                            pos_team = 1 if pos % 2 == 0 else 2

                            if calling_team == pos_team:  # This position's team called
                                if winning_team == pos_team:
                                    # Team won
                                    if points_awarded == 2:
                                        hand_reward = 0.25  # March
                                    else:
                                        hand_reward = 0.10  # Made it
                                else:
                                    # Got euchred
                                    hand_reward = -0.30
                                    episode.euchres[pos] += 1
                            elif calling_team is not None:  # Opponent called
                                if winning_team == pos_team:
                                    # Successfully defended
                                    hand_reward = 0.30  # Euchred opponent
                                else:
                                    # Opponent made it
                                    hand_reward = -0.15
                            else:
                                hand_reward = 0.0

                            episode.hand_rewards[pos].append(hand_reward)

                        # Reset hand info
                        current_hand_info = {
                            "caller_position": None,
                            "calling_team": None,
                            "calling_trump_suit_char": None,
                            "calling_hand_cards": None,
                            "tricks_won_by_pos": {0: 0, 1: 0, 2: 0, 3: 0},
                            "was_forced_call": False,
                        }

                        # Start new hand if game not over
                        if game.state.phase != GamePhase.GAME_OVER:
                            game.start_new_hand()

            elif game.state.phase == GamePhase.HAND_COMPLETE:
                game.start_new_hand()

        # GAME-LEVEL REWARD (score differential from each position's perspective)
        episode.team1_score = game.state.team1_score
        episode.team2_score = game.state.team2_score

        # Team 1 perspective (positions 0, 2)
        team1_reward = (game.state.team1_score - game.state.team2_score) / 10.0
        # Team 2 perspective (positions 1, 3)
        team2_reward = (game.state.team2_score - game.state.team1_score) / 10.0

        episode.game_reward = team1_reward  # For statistics (team 1 perspective)

        # Track statistics
        for pos in range(4):
            episode.tricks_won[pos] = sum(
                current_hand_info["tricks_won_by_pos"].get(pos, 0) for _ in range(1)
            )

        return episode

    def compute_returns(self, rewards: List[float], gamma: float = 0.99) -> List[float]:
        """
        Compute discounted returns for a sequence of rewards.

        Args:
            rewards: List of rewards
            gamma: Discount factor

        Returns:
            List of discounted returns
        """
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        return returns

    def train_on_batch(self, episodes: List[Episode]) -> Dict[str, float]:
        """
        Train the model on a batch of episodes using policy gradient.
        Now processes all 4 positions with their own perspective.

        Args:
            episodes: List of Episode objects

        Returns:
            Dictionary with training statistics
        """
        # Collect all decisions and their returns
        card_states = []
        card_actions = []
        card_log_probs = []
        card_entropies = []
        card_returns = []

        trump_log_probs = []
        trump_entropies = []
        trump_returns = []

        discard_log_probs = []
        discard_entropies = []
        discard_returns = []

        total_reward = 0.0
        total_games_played = len(episodes)

        for episode in episodes:
            # Compute total reward for this episode (from team 1 perspective for stats)
            game_reward = episode.game_reward
            total_reward += game_reward

            # Compute rewards for ALL 4 positions from their own perspective
            all_rewards_by_pos = {}
            for pos in range(4):
                pos_team = 1 if pos % 2 == 0 else 2
                # Game reward from this position's perspective
                if pos_team == 1:
                    pos_game_reward = (episode.team1_score - episode.team2_score) / 10.0
                else:
                    pos_game_reward = (episode.team2_score - episode.team1_score) / 10.0

                all_rewards_by_pos[pos] = (
                    list(episode.trick_rewards[pos])
                    + list(episode.hand_rewards[pos])
                    + [pos_game_reward]
                )

            # Process card decisions for all positions
            for state, action, log_prob, pos in episode.card_decisions:
                rewards = all_rewards_by_pos[pos]
                returns = self.compute_returns(rewards, self.gamma)

            for i, (state, action, log_prob, entropy, pos) in enumerate(episode.card_decisions):
                card_log_probs.append(log_prob)
                card_entropies.append(entropy)
                card_returns.append(card_ret[i] if i < len(card_ret) else game_reward)

            # Process trump decisions for all positions
            for state, action, log_prob, pos in episode.trump_decisions:
                rewards = all_rewards_by_pos[pos]
                returns = self.compute_returns(rewards, self.gamma)

            for i, (state, action, log_prob, entropy, pos) in enumerate(episode.trump_decisions):
                trump_log_probs.append(log_prob)
                trump_entropies.append(entropy)
                trump_returns.append(trump_ret[i] if i < len(trump_ret) else game_reward)

            # Process discard decisions for all positions
            for state, action, log_prob, pos in episode.discard_decisions:
                rewards = all_rewards_by_pos[pos]
                returns = self.compute_returns(rewards, self.gamma)

            for i, (state, action, log_prob, entropy, pos) in enumerate(episode.discard_decisions):
                discard_log_probs.append(log_prob)
                discard_entropies.append(entropy)
                discard_returns.append(discard_ret[i] if i < len(discard_ret) else game_reward)

        # Update running reward baseline
        if self.running_reward is None:
            self.running_reward = total_reward / len(episodes)
        else:
            self.running_reward = 0.95 * self.running_reward + 0.05 * (
                total_reward / len(episodes)
            )

        # Compute policy gradient loss with per-type baselines and entropy
        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        total_entropy = torch.tensor(0.0, device=self.device)

        critic_loss = torch.tensor(0.0, device=self.device)

        def _compute_advantages_and_loss(log_probs, entropies, returns, states, decision_type):
            """Compute policy loss + critic loss for one decision type."""
            nonlocal total_loss, total_entropy, critic_loss

            lp = torch.stack([l.squeeze() if l.dim() > 0 else l for l in log_probs])
            ret_t = torch.FloatTensor(returns).to(self.device)

            # If critic available, use V(s) as baseline instead of mean(returns)
            if self.critic is not None and states:
                state_batch = torch.FloatTensor(np.array(states)).to(self.device)
                if decision_type == "card":
                    values = self.critic.forward_card(state_batch).squeeze()
                elif decision_type == "trump":
                    values = self.critic.forward_trump(state_batch).squeeze()
                else:
                    values = self.critic.forward_discard(state_batch).squeeze()
                advantages = ret_t - values.detach()
                # Critic loss: MSE between V(s) and actual returns
                critic_loss = critic_loss + nn.functional.mse_loss(values, ret_t)
            else:
                advantages = ret_t - ret_t.mean()

            # Normalize advantages for stable training
            if advantages.numel() > 1:
                std = advantages.std()
                if std > 1e-8:
                    advantages = advantages / std

            policy_loss = -(lp * advantages).mean()
            total_loss = total_loss + policy_loss

            ent = torch.stack([e.squeeze() if e.dim() > 0 else e for e in entropies])
            total_entropy = total_entropy + ent.mean()

        if card_log_probs:
            card_states = [s for s, _, _, _, _ in
                          [d for ep in episodes for d in ep.card_decisions]]
            _compute_advantages_and_loss(
                card_log_probs, card_entropies, card_returns, card_states, "card"
            )

        if trump_log_probs:
            trump_states = [s for s, _, _, _, _ in
                           [d for ep in episodes for d in ep.trump_decisions]]
            _compute_advantages_and_loss(
                trump_log_probs, trump_entropies, trump_returns, trump_states, "trump"
            )

        if discard_log_probs:
            discard_states = [s for s, _, _, _, _ in
                             [d for ep in episodes for d in ep.discard_decisions]]
            _compute_advantages_and_loss(
                discard_log_probs, discard_entropies, discard_returns, discard_states, "discard"
            )

        # Entropy bonus: encourages exploration, prevents policy collapse
        total_loss = total_loss - self.entropy_beta * total_entropy

        # Add critic loss if using actor-critic
        if self.critic is not None:
            total_loss = total_loss + 0.5 * critic_loss

        # Backpropagation
        self.optimizer.zero_grad()
        if self.critic_optimizer is not None:
            self.critic_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        if self.critic is not None:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.optimizer.step()
        if self.critic_optimizer is not None:
            self.critic_optimizer.step()

        # Update self-play opponent periodically
        if self.self_play:
            self.updates_since_opponent_refresh += 1
            if self.updates_since_opponent_refresh >= self.opponent_update_interval:
                self.update_opponent()

        # Update statistics
        self.total_games += len(episodes)
        self.avg_reward = total_reward / len(episodes)

        return {
            "loss": total_loss.item(),
            "avg_reward": self.avg_reward,
            "avg_score_diff": self.avg_reward,  # Average score differential (self-play)
            "running_reward": self.running_reward,
            "entropy": total_entropy.item(),
            "critic_loss": critic_loss.item() if self.critic is not None else 0.0,
        }
