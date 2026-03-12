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


PASSED_PENALTY = 0.0  # No penalty for passing out. This makes passing a safe baseline.


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
        self.total_hands = 0
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

    def update_opponent(self):
        """Deep copy current model weights into opponent model and add to pool."""
        self.opponent_model = copy.deepcopy(self.model)
        self.opponent_model.eval()

        # Add snapshot to opponent pool, keep max 10
        snapshot = copy.deepcopy(self.model)
        snapshot.eval()
        self.opponent_pool.append(snapshot)
        if len(self.opponent_pool) > 10:
            self.opponent_pool.pop(0)

    def select_action_with_exploration(
        self, state_encoding: np.ndarray, valid_indices: List[int], decision_type: str
    ) -> Tuple[int, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Select action using model's policy with epsilon-greedy exploration.
        Uses logits from the network (not softmax output) to avoid double-softmax.
        Also queries the critic for V(s) if available.

        Returns:
            (action_index, log_probability, entropy, value_estimate_or_None)
        """
        state_tensor = torch.FloatTensor(state_encoding).unsqueeze(0).to(self.device)

        # Get logits (not probabilities) from network
        if decision_type == "card":
            logits = self.model.forward(state_tensor, return_logits=True)
        elif decision_type == "trump":
            logits = self.model.forward_trump(state_tensor, return_logits=True)
        else:  # discard
            logits = self.model.forward_discard(state_tensor, return_logits=True)

        mask = torch.full_like(logits, float("-inf"))
        mask[0, valid_indices] = 0
        masked_logits = logits + mask

        dist = torch.distributions.Categorical(logits=masked_logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        # Get value estimate from critic if available
        value = None
        if self.critic is not None:
            with torch.no_grad():
                if decision_type == "card":
                    value = self.critic.forward_card(state_tensor).squeeze()
                elif decision_type == "trump":
                    value = self.critic.forward_trump(state_tensor).squeeze()
                else:
                    value = self.critic.forward_discard(state_tensor).squeeze()

        return action.item(), log_prob, entropy, value

    def _get_opponent_model(self):
        """Get the opponent model for self-play inference."""
        if not self.opponent_pool:
            return self.opponent_model
        # Randomly pick from pool for diversity
        return random.choice(self.opponent_pool)

    def _opponent_select_action(
        self, opponent_model, state_encoding: np.ndarray, valid_indices: List[int], decision_type: str
    ) -> int:
        """Select action using opponent model (no gradient tracking)."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state_encoding).unsqueeze(0).to(self.device)
            if decision_type == "card":
                logits = opponent_model.forward(state_tensor, return_logits=True)
            elif decision_type == "trump":
                logits = opponent_model.forward_trump(state_tensor, return_logits=True)
            else:
                logits = opponent_model.forward_discard(state_tensor, return_logits=True)

            mask = torch.full_like(logits, float("-inf"))
            mask[0, valid_indices] = 0
            masked_logits = logits + mask
            dist = torch.distributions.Categorical(logits=masked_logits)
            return dist.sample().item()

    def play_game(self, opponent_type: str = "passive") -> Episode:
        """
        Play a single game.

        Args:
            opponent_type:
              "passive"     - Team 2 always passes trump and plays random cards.
              "self_play"   - Team 2 uses frozen opponent model for everything.
              "semi_passive"- Team 2 always passes trump (model team is always
                              the only caller) but uses frozen model for card play.
                              Best of both worlds: maintains calling incentive while
                              training card play against competent opponents.
        """
        episode = Episode()
        game = EuchreGame(f"training-{random.randint(1000, 9999)}")

        for i in range(4):
            game.add_player(f"Player{i}", PlayerType.RANDOM_AI)

        # Get opponent model for self-play / semi-passive
        opponent_model = None
        if opponent_type in ("self_play", "semi_passive") and self.opponent_model is not None:
            opponent_model = self._get_opponent_model()

        current_hand_info = {
            "caller_position": None,
            "calling_team": None,
        }

        game.start_new_hand()

        while game.state.phase != GamePhase.GAME_OVER:
            current_pos = game.state.current_player_position
            is_model_player = current_pos in [0, 2]
            is_opponent_card_selfplay = (opponent_type in ("self_play", "semi_passive")) and (current_pos in [1, 3]) and (opponent_model is not None)
            is_opponent_trump_selfplay = (opponent_type == "self_play") and (current_pos in [1, 3]) and (opponent_model is not None)

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

                if is_model_player:
                    decision_idx, log_prob, entropy, value = self.select_action_with_exploration(
                        trump_state, valid_indices, "trump"
                    )
                    episode.events[current_pos].append(
                        {
                            "type": "decision",
                            "decision_type": "trump",
                            "state": trump_state,
                            "action": decision_idx,
                            "log_prob": log_prob,
                            "entropy": entropy,
                            "value": value,
                        }
                    )
                elif is_opponent_trump_selfplay:
                    # Full self-play: frozen model decides trump
                    decision_idx = self._opponent_select_action(
                        opponent_model, trump_state, valid_indices, "trump"
                    )
                else:
                    # Passive / semi-passive: opponent always passes trump
                    decision_idx = 4

                if decision_idx == 4:  # Pass
                    result = game.pass_trump()
                    if result == "hand_over":
                        episode.total_hands += 1
                        for pos in range(4):
                            episode.events[pos].append(
                                {"type": "reward", "value": PASSED_PENALTY}
                            )
                        if game.state.phase != GamePhase.GAME_OVER:
                            game.start_new_hand()
                else:  # Call
                    suit_map = [Suit.CLUBS, Suit.DIAMONDS, Suit.HEARTS, Suit.SPADES]
                    if game.state.phase == GamePhase.TRUMP_SELECTION_ROUND2:
                        selected_suit = suit_map[decision_idx]
                    else:
                        # In Round 1, we must call the turned up suit
                        selected_suit = (
                            game.state.turned_up_card.suit
                            if game.state.turned_up_card
                            else Suit.CLUBS
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
                is_dealer_model = dealer_pos in [0, 2]
                is_dealer_opponent_selfplay = (opponent_type in ("self_play", "semi_passive")) and (dealer_pos in [1, 3]) and (opponent_model is not None)

                if is_dealer_model:
                    game_state_dict = game.get_state(perspective_position=dealer_pos)
                    hand_with_pickup = [str(card) for card in dealer.hand]
                    discard_state = encode_discard_state(
                        game_state_dict, hand_with_pickup
                    )

                    valid_indices = [
                        self.all_cards.index(str(c))
                        for c in dealer.hand
                        if str(c) in self.all_cards
                    ]
                    if not valid_indices:
                        valid_indices = [0]

                    decision_idx, log_prob, entropy, value = self.select_action_with_exploration(
                        discard_state, valid_indices, "discard"
                    )
                    episode.events[dealer_pos].append(
                        {
                            "type": "decision",
                            "decision_type": "discard",
                            "state": discard_state,
                            "action": decision_idx,
                            "log_prob": log_prob,
                            "entropy": entropy,
                            "value": value,
                        }
                    )
                    card_to_discard = next(
                        (
                            c
                            for c in dealer.hand
                            if str(c) == self.all_cards[decision_idx]
                        ),
                        dealer.hand[0],
                    )
                elif is_dealer_opponent_selfplay:
                    # Self-play opponent uses frozen model for discard
                    game_state_dict = game.get_state(perspective_position=dealer_pos)
                    hand_with_pickup = [str(card) for card in dealer.hand]
                    discard_state = encode_discard_state(
                        game_state_dict, hand_with_pickup
                    )
                    valid_indices = [
                        self.all_cards.index(str(c))
                        for c in dealer.hand
                        if str(c) in self.all_cards
                    ]
                    if not valid_indices:
                        valid_indices = [0]
                    decision_idx = self._opponent_select_action(
                        opponent_model, discard_state, valid_indices, "discard"
                    )
                    card_to_discard = next(
                        (c for c in dealer.hand if str(c) == self.all_cards[decision_idx]),
                        dealer.hand[0],
                    )
                else:
                    # Passive opponent discards randomly
                    card_to_discard = random.choice(dealer.hand)

                game.dealer_discard(card_to_discard)

            elif game.state.phase == GamePhase.PLAYING:
                valid_cards = game.get_valid_moves(current_pos)

                if is_model_player:
                    game_state_dict = game.get_state(perspective_position=current_pos)
                    state_encoding = encode_game_state(game_state_dict)
                    valid_indices = [
                        self.all_cards.index(str(c))
                        for c in valid_cards
                        if str(c) in self.all_cards
                    ]

                    if not valid_indices:
                        valid_indices = [0]

                    decision_idx, log_prob, entropy, value = self.select_action_with_exploration(
                        state_encoding, valid_indices, "card"
                    )
                    episode.events[current_pos].append(
                        {
                            "type": "decision",
                            "decision_type": "card",
                            "state": state_encoding,
                            "action": decision_idx,
                            "log_prob": log_prob,
                            "entropy": entropy,
                            "value": value,
                        }
                    )

                    selected_card = next(
                        (
                            c
                            for c in valid_cards
                            if str(c) == self.all_cards[decision_idx]
                        ),
                        valid_cards[0],
                    )
                elif is_opponent_card_selfplay:
                    # Self-play / semi-passive: frozen model plays cards
                    game_state_dict = game.get_state(perspective_position=current_pos)
                    state_encoding = encode_game_state(game_state_dict)
                    valid_indices = [
                        self.all_cards.index(str(c))
                        for c in valid_cards
                        if str(c) in self.all_cards
                    ]
                    if not valid_indices:
                        valid_indices = [0]
                    opp_decision_idx = self._opponent_select_action(
                        opponent_model, state_encoding, valid_indices, "card"
                    )
                    selected_card = next(
                        (c for c in valid_cards if str(c) == self.all_cards[opp_decision_idx]),
                        valid_cards[0],
                    )
                else:
                    # Passive opponent plays randomly
                    selected_card = random.choice(valid_cards)

                result = game.play_card(selected_card)

                if result.get("trick_complete"):
                    winner_pos = result["trick_winner"]
                    episode.tricks_won[winner_pos] += 1
                    for pos in range(4):
                        # Symmetric trick rewards to avoid "participation trophy" bias
                        reward = 0.05 if (pos % 2) == (winner_pos % 2) else -0.05
                        episode.events[pos].append({"type": "reward", "value": reward})

                if result.get("hand_complete"):
                    episode.total_hands += 1
                    hand_winner = result["hand_winner"]
                    winning_team = hand_winner["winning_team"]
                    points = hand_winner["points_awarded"]
                    calling_team = current_hand_info["calling_team"]

                    for pos in range(4):
                        pos_team = 1 if pos % 2 == 0 else 2
                        if calling_team == pos_team:
                            # CALLER REWARDS: Reward march generously, reduce euchre penalty.
                            # Old: success=+1.5, euchre=-2.0 → break-even at 57% win rate (too high).
                            # New: march=+2.0, win=+1.0, euchre=-1.2 → break-even at 45% win rate.
                            if winning_team == pos_team:
                                reward = 2.0 if points >= 2 else 1.0
                            else:
                                reward = -1.2
                            if winning_team != pos_team and pos == current_hand_info["caller_position"]:
                                episode.euchres[pos] += 1
                        elif calling_team is not None:
                            # DEFENDER REWARDS: High reward for success (Euchre), moderate penalty for failure
                            # Making defense rewarding encourages passing with good defensive hands.
                            reward = 1.0 if winning_team == pos_team else -0.5
                        else:
                            reward = PASSED_PENALTY
                        episode.events[pos].append({"type": "reward", "value": reward})
                        # Mark hand boundary so returns don't bleed across independent hands
                        episode.events[pos].append({"type": "hand_boundary"})

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
        Fixes applied:
        - Returns reset at hand boundaries (hands are independent episodes)
        - Entropy bonus applied to loss to prevent policy collapse
        - Critic used for advantage computation when available
        """
        all_log_probs = {"card": [], "trump": [], "discard": []}
        all_returns = {"card": [], "trump": [], "discard": []}
        all_entropies = {"card": [], "trump": [], "discard": []}
        all_values = {"card": [], "trump": [], "discard": []}   # V(s) from critic
        total_reward = 0.0

        # Statistics for Team 1 (Model)
        total_wins = 0
        total_calls = 0
        total_euchres = 0
        total_hands = 0

        for episode in episodes:
            total_reward += episode.game_reward
            total_hands += episode.total_hands

            # Team 1 stats
            if episode.team1_score > episode.team2_score:
                total_wins += 1

            total_calls += episode.trump_calls[0] + episode.trump_calls[2]
            total_euchres += episode.euchres[0] + episode.euchres[2]

            for pos in range(4):
                events = episode.events[pos]
                # Calculate returns backwards, resetting at hand boundaries
                # Hands are independent episodes — reward from hand N should not
                # propagate back into decisions from hand N-1.
                returns = []
                G = 0
                for event in reversed(events):
                    if event["type"] == "hand_boundary":
                        G = 0  # Reset return at hand boundary
                    elif event["type"] == "reward":
                        G = event["value"] + self.gamma * G
                    elif event["type"] == "decision":
                        returns.insert(0, G)

                # Match returns, entropies, and values to decisions
                decision_idx = 0
                for event in events:
                    if event["type"] == "decision":
                        dtype = event["decision_type"]
                        all_log_probs[dtype].append(event["log_prob"])
                        all_returns[dtype].append(returns[decision_idx])
                        all_entropies[dtype].append(event["entropy"])
                        if event.get("value") is not None:
                            all_values[dtype].append(event["value"])
                        decision_idx += 1

        if self.running_reward is None:
            self.running_reward = total_reward / len(episodes)
        else:
            self.running_reward = 0.95 * self.running_reward + 0.05 * (
                total_reward / len(episodes)
            )

        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        total_entropy = torch.tensor(0.0, device=self.device)
        critic_loss = torch.tensor(0.0, device=self.device)

        # Trump head gets ~5x fewer updates than card head, so upweight its entropy
        # to prevent it from collapsing to "always pass" before it can learn.
        entropy_weights = {"card": 1.0, "trump": 5.0, "discard": 1.0}

        for dtype in ["card", "trump", "discard"]:
            if not all_log_probs[dtype]:
                continue

            log_probs = torch.stack(
                [lp.squeeze() if lp.dim() > 0 else lp for lp in all_log_probs[dtype]]
            )
            entropies = torch.stack(
                [e.squeeze() if e.dim() > 0 else e for e in all_entropies[dtype]]
            )
            returns = torch.FloatTensor(all_returns[dtype]).to(self.device)

            # Use critic advantage if available, otherwise standardize raw returns
            if self.critic is not None and len(all_values[dtype]) == len(all_returns[dtype]):
                values = torch.stack(
                    [v.squeeze() if v.dim() > 0 else v for v in all_values[dtype]]
                ).detach()
                advantages = returns - values
                # Critic loss: MSE between predicted values and actual returns
                values_for_loss = torch.stack(
                    [v.squeeze() if v.dim() > 0 else v for v in all_values[dtype]]
                )
                critic_loss = critic_loss + nn.functional.mse_loss(values_for_loss, returns)
            else:
                advantages = returns

            # Standardize advantages for training stability
            if len(advantages) > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Policy gradient loss using advantages
            policy_loss = -(log_probs * advantages).mean()
            entropy_bonus = entropies.mean() * entropy_weights[dtype]
            total_entropy = total_entropy + entropy_bonus
            total_loss = total_loss + policy_loss

        # Subtract entropy bonus and add critic loss
        total_loss = total_loss - self.entropy_beta * total_entropy
        if self.critic is not None:
            total_loss = total_loss + 0.5 * critic_loss

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
                self.updates_since_opponent_refresh = 0

        self.total_games += len(episodes)
        self.avg_reward = total_reward / len(episodes)

        # Calculate final stats
        win_rate = total_wins / len(episodes)
        call_rate = total_calls / max(1, total_hands)
        call_success_rate = (total_calls - total_euchres) / max(1, total_calls)

        return {
            "loss": total_loss.item(),
            "avg_reward": self.avg_reward,
            "avg_score_diff": self.avg_reward,
            "running_reward": self.running_reward,
            "win_rate": win_rate,
            "call_rate": call_rate,
            "call_success_rate": call_success_rate,
            "entropy": total_entropy.item(),
        }
