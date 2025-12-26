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


PASSED_PENALTY = -1.0  # Penalty for passing (increased to force active play)


class Episode:
    """Records a single game episode with all decisions and rewards"""

    def __init__(self):
        # Store chronological events for each position
        # Each event is a dict:
        # {'type': 'decision', 'decision_type': 'card'|'trump'|'discard', 'state': ..., 'action': ..., 'log_prob': ...}
        # OR {'type': 'reward', 'value': float}
        self.events = defaultdict(list)

        # Track game statistics
        self.team1_score = 0
        self.team2_score = 0
        self.tricks_won = defaultdict(int)  # position -> count
        self.trump_calls = defaultdict(int)  # position -> count
        self.euchres = defaultdict(int)  # position -> count
        self.game_reward = 0.0  # For statistics (team 1 perspective)


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

    def select_action_with_exploration(
        self, state_encoding: np.ndarray, valid_indices: List[int], decision_type: str
    ) -> Tuple[int, torch.Tensor]:
        """
        Select action using model's policy with epsilon-greedy exploration.
        """
        # Epsilon-greedy exploration
        if random.random() < self.exploration_rate:
            action_idx = random.choice(valid_indices)
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

                mask = torch.full_like(probs, float("-inf"))
                mask[0, valid_indices] = 0
                masked_probs = torch.softmax(probs + mask, dim=1)
                log_prob = torch.log(masked_probs[0, action_idx] + 1e-10)

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

        mask = torch.full_like(probs, float("-inf"))
        mask[0, valid_indices] = 0
        masked_probs = torch.softmax(probs + mask, dim=1)

        dist = torch.distributions.Categorical(masked_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

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
        """
        episode = Episode()
        game = EuchreGame(f"training-{random.randint(1000, 9999)}")

        for i in range(4):
            game.add_player(f"Player{i}", PlayerType.RANDOM_AI)

        current_hand_info = {
            "caller_position": None,
            "calling_team": None,
        }

        game.start_new_hand()

        while game.state.phase != GamePhase.GAME_OVER:
            current_pos = game.state.current_player_position

            if game.state.phase in [
                GamePhase.TRUMP_SELECTION_ROUND1,
                GamePhase.TRUMP_SELECTION_ROUND2,
            ]:
                game_state_dict = game.get_state(perspective_position=current_pos)
                turned_up_card = (
                    str(game.state.turned_up_card)
                    if game.state.turned_up_card
                    else None
                )
                trump_state = encode_trump_state(game_state_dict, turned_up_card)

                # Determine valid options
                if game.state.phase == GamePhase.TRUMP_SELECTION_ROUND1:
                    suit_map_enum = {
                        Suit.CLUBS: 0,
                        Suit.DIAMONDS: 1,
                        Suit.HEARTS: 2,
                        Suit.SPADES: 3,
                    }
                    turned_up_suit_idx = (
                        suit_map_enum.get(game.state.turned_up_card.suit)
                        if game.state.turned_up_card
                        else None
                    )
                    valid_indices = (
                        [turned_up_suit_idx, 4]
                        if turned_up_suit_idx is not None
                        else [4]
                    )
                else:
                    turned_up_suit = (
                        game.state.turned_up_card.suit
                        if game.state.turned_up_card
                        else Suit.CLUBS
                    )
                    suit_map_enum = {
                        Suit.CLUBS: 0,
                        Suit.DIAMONDS: 1,
                        Suit.HEARTS: 2,
                        Suit.SPADES: 3,
                    }
                    turned_up_suit_idx = suit_map_enum.get(turned_up_suit, 0)
                    valid_indices = [i for i in range(4) if i != turned_up_suit_idx]
                    if current_pos != game.state.dealer_position:
                        valid_indices.append(4)

                decision_idx, log_prob = self.select_action_with_exploration(
                    trump_state, valid_indices, "trump"
                )
                episode.events[current_pos].append(
                    {
                        "type": "decision",
                        "decision_type": "trump",
                        "state": trump_state,
                        "action": decision_idx,
                        "log_prob": log_prob,
                    }
                )

                if decision_idx == 4:  # Pass
                    result = game.pass_trump()
                    if result == "hand_over":
                        for pos in range(4):
                            episode.events[pos].append(
                                {"type": "reward", "value": PASSED_PENALTY}
                            )
                        if game.state.phase != GamePhase.GAME_OVER:
                            game.start_new_hand()
                else:  # Call
                    suit_map = [Suit.CLUBS, Suit.DIAMONDS, Suit.HEARTS, Suit.SPADES]
                    selected_suit = (
                        suit_map[decision_idx]
                        if game.state.phase == GamePhase.TRUMP_SELECTION_ROUND2
                        else game.state.turned_up_card.suit
                    )
                    game.call_trump(selected_suit)
                    current_hand_info["caller_position"] = current_pos
                    current_hand_info["calling_team"] = (
                        1 if current_pos in [0, 2] else 2
                    )
                    episode.trump_calls[current_pos] += 1

            elif game.state.phase == GamePhase.DEALER_DISCARD:
                dealer_pos = game.state.dealer_position
                dealer = game.state.get_player(dealer_pos)
                game_state_dict = game.get_state(perspective_position=dealer_pos)
                hand_with_pickup = [str(card) for card in dealer.hand]
                discard_state = encode_discard_state(game_state_dict, hand_with_pickup)

                valid_indices = [
                    self.all_cards.index(str(c))
                    for c in dealer.hand
                    if str(c) in self.all_cards
                ]
                if not valid_indices:
                    valid_indices = [0]

                decision_idx, log_prob = self.select_action_with_exploration(
                    discard_state, valid_indices, "discard"
                )
                episode.events[dealer_pos].append(
                    {
                        "type": "decision",
                        "decision_type": "discard",
                        "state": discard_state,
                        "action": decision_idx,
                        "log_prob": log_prob,
                    }
                )

                card_to_discard = next(
                    (c for c in dealer.hand if str(c) == self.all_cards[decision_idx]),
                    dealer.hand[0],
                )
                game.dealer_discard(card_to_discard)

            elif game.state.phase == GamePhase.PLAYING:
                game_state_dict = game.get_state(perspective_position=current_pos)
                state_encoding = encode_game_state(game_state_dict)
                valid_cards = game.get_valid_moves(current_pos)
                valid_indices = [
                    self.all_cards.index(str(c))
                    for c in valid_cards
                    if str(c) in self.all_cards
                ]

                if not valid_indices:
                    valid_indices = [0]

                decision_idx, log_prob = self.select_action_with_exploration(
                    state_encoding, valid_indices, "card"
                )
                episode.events[current_pos].append(
                    {
                        "type": "decision",
                        "decision_type": "card",
                        "state": state_encoding,
                        "action": decision_idx,
                        "log_prob": log_prob,
                    }
                )

                selected_card = next(
                    (c for c in valid_cards if str(c) == self.all_cards[decision_idx]),
                    valid_cards[0],
                )
                result = game.play_card(selected_card)

                if result.get("trick_complete"):
                    winner_pos = result["trick_winner"]
                    episode.tricks_won[winner_pos] += 1
                    for pos in range(4):
                        reward = (
                            0.05 if (pos % 2) == (winner_pos % 2) else -0.02
                        )  # Reward winning tricks
                        episode.events[pos].append({"type": "reward", "value": reward})

                if result.get("hand_complete"):
                    hand_winner = result["hand_winner"]
                    winning_team = hand_winner["winning_team"]
                    points = hand_winner["points_awarded"]
                    calling_team = current_hand_info["calling_team"]

                    for pos in range(4):
                        pos_team = 1 if pos % 2 == 0 else 2
                        if calling_team == pos_team:
                            # CALLER REWARDS: High reward for success, moderate penalty for failure
                            reward = (
                                (1.5 if points >= 2 else 1.0)
                                if winning_team == pos_team
                                else -0.5
                            )
                            if winning_team != pos_team:
                                episode.euchres[pos] += 1
                        elif calling_team is not None:
                            # DEFENDER REWARDS: Moderate reward for success, moderate penalty for failure
                            reward = 0.5 if winning_team == pos_team else -0.5
                        else:
                            reward = PASSED_PENALTY
                        episode.events[pos].append({"type": "reward", "value": reward})

                    current_hand_info = {"caller_position": None, "calling_team": None}
                    if game.state.phase != GamePhase.GAME_OVER:
                        game.start_new_hand()

            elif game.state.phase == GamePhase.HAND_COMPLETE:
                game.start_new_hand()

        # Final game rewards
        episode.team1_score = game.state.team1_score
        episode.team2_score = game.state.team2_score
        for pos in range(4):
            pos_team = 1 if pos % 2 == 0 else 2
            score_diff = (
                (game.state.team1_score - game.state.team2_score)
                if pos_team == 1
                else (game.state.team2_score - game.state.team1_score)
            )
            episode.events[pos].append({"type": "reward", "value": score_diff / 5.0})

        episode.game_reward = (game.state.team1_score - game.state.team2_score) / 10.0
        return episode

    def train_on_batch(self, episodes: List[Episode]) -> Dict[str, float]:
        """
        Train the model on a batch of episodes using REINFORCE with proper credit assignment.
        """
        all_log_probs = {"card": [], "trump": [], "discard": []}
        all_returns = {"card": [], "trump": [], "discard": []}
        total_reward = 0.0

        for episode in episodes:
            total_reward += episode.game_reward
            for pos in range(4):
                events = episode.events[pos]
                # Calculate returns backwards
                returns = []
                G = 0
                for event in reversed(events):
                    if event["type"] == "reward":
                        G = event["value"] + self.gamma * G
                    else:
                        returns.insert(0, G)

                # Match returns to decisions
                decision_idx = 0
                for event in events:
                    if event["type"] == "decision":
                        dtype = event["decision_type"]
                        all_log_probs[dtype].append(event["log_prob"])
                        all_returns[dtype].append(returns[decision_idx])
                        decision_idx += 1

        if self.running_reward is None:
            self.running_reward = total_reward / len(episodes)
        else:
            self.running_reward = 0.95 * self.running_reward + 0.05 * (
                total_reward / len(episodes)
            )

        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)

        for dtype in ["card", "trump", "discard"]:
            if all_log_probs[dtype]:
                log_probs = torch.stack(
                    [
                        lp.squeeze() if lp.dim() > 0 else lp
                        for lp in all_log_probs[dtype]
                    ]
                )
                returns = torch.FloatTensor(all_returns[dtype]).to(self.device)
                # Standardize returns for stability
                if len(returns) > 1:
                    returns = (returns - returns.mean()) / (returns.std() + 1e-8)

                loss = -(log_probs * returns).mean()
                total_loss = total_loss + loss

        self.optimizer.zero_grad()
        if self.critic_optimizer is not None:
            self.critic_optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        if self.critic_optimizer is not None:
            self.critic_optimizer.step()

        # Update self-play opponent periodically
        if self.self_play:
            self.updates_since_opponent_refresh += 1
            if self.updates_since_opponent_refresh >= self.opponent_update_interval:
                self.update_opponent()

        self.total_games += len(episodes)
        self.avg_reward = total_reward / len(episodes)

        return {
            "loss": total_loss.item(),
            "avg_reward": self.avg_reward,
            "avg_score_diff": self.avg_reward,
            "running_reward": self.running_reward,
            "entropy": total_entropy.item(),
            "critic_loss": critic_loss.item() if self.critic is not None else 0.0,
        }
