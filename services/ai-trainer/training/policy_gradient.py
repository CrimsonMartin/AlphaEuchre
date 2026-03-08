"""
Policy Gradient Trainer for Euchre AI
Uses REINFORCE algorithm with reward shaping for active play
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
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
    """Trains a single model using policy gradient (REINFORCE)"""

    def __init__(
        self,
        model: BasicEuchreNN,
        learning_rate: float = 0.0003,
        gamma: float = 0.99,
        entropy_beta: float = 0.01,
        exploration_rate: float = 0.1,
        use_cuda: bool = True,
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
        Play a single game with the model as partners (positions 0 & 2)
        and random AI as opponents (positions 1 & 3).

        Per-decision rewards are tracked so each decision gets credit for
        outcomes it directly influenced (not a flat global return).
        """
        episode = Episode()

        # Create game
        game = EuchreGame(f"training-{random.randint(1000, 9999)}")

        # Add players
        for i in range(4):
            game.add_player(f"Player{i}", PlayerType.RANDOM_AI)

        # Track hand-level info
        current_hand_info = {
            "caller_position": None,
            "calling_team": None,
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
            is_model_position = current_pos in [0, 2]

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

                if is_model_position:
                    decision_idx, log_prob, entropy = self.select_action_with_exploration(
                        trump_state, valid_trump_indices, "trump"
                    )
                    episode.trump_decisions.append(
                        (trump_state, decision_idx, log_prob, entropy, current_pos)
                    )
                    # Trump decisions get 0 immediate reward; hand outcome
                    # reward is assigned retroactively when the hand ends
                    episode.trump_rewards.append(0.0)
                else:
                    decision_idx = random.choice(valid_trump_indices)

                # Execute decision
                if decision_idx == 4:
                    try:
                        game.pass_trump()
                    except:
                        if game.state.turned_up_card:
                            game.call_trump(game.state.turned_up_card.suit)
                            current_hand_info["caller_position"] = current_pos
                            current_hand_info["calling_team"] = (
                                1 if current_pos in [0, 2] else 2
                            )
                            if is_model_position:
                                episode.trump_calls[current_pos] += 1
                else:
                    try:
                        if game.state.turned_up_card:
                            game.call_trump(game.state.turned_up_card.suit)
                            current_hand_info["caller_position"] = current_pos
                            current_hand_info["calling_team"] = (
                                1 if current_pos in [0, 2] else 2
                            )
                            if is_model_position:
                                episode.trump_calls[current_pos] += 1
                    except:
                        try:
                            game.pass_trump()
                        except:
                            if game.state.turned_up_card:
                                game.call_trump(game.state.turned_up_card.suit)
                                current_hand_info["caller_position"] = current_pos
                                current_hand_info["calling_team"] = (
                                    1 if current_pos in [0, 2] else 2
                                )
                                if is_model_position:
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

                if is_model_position:
                    decision_idx, log_prob, entropy = self.select_action_with_exploration(
                        trump_state, valid_trump_indices, "trump"
                    )
                    episode.trump_decisions.append(
                        (trump_state, decision_idx, log_prob, entropy, current_pos)
                    )
                    episode.trump_rewards.append(0.0)
                else:
                    decision_idx = random.choice(valid_trump_indices)

                # Execute decision
                if decision_idx == 4:
                    if is_dealer:
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
                        current_hand_info["was_forced_call"] = True
                        if is_model_position:
                            episode.trump_calls[current_pos] += 1
                    else:
                        try:
                            game.pass_trump()
                        except:
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
                            if is_model_position:
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
                        if is_model_position:
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
                            current_hand_info["was_forced_call"] = True
                            if is_model_position:
                                episode.trump_calls[current_pos] += 1
                        else:
                            try:
                                game.pass_trump()
                            except:
                                game.call_trump(random.choice(available_suits))
                                current_hand_info["caller_position"] = current_pos
                                current_hand_info["calling_team"] = (
                                    1 if current_pos in [0, 2] else 2
                                )
                                if is_model_position:
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

                if is_model_position:
                    if valid_discard_indices:
                        card_idx, log_prob, entropy = self.select_action_with_exploration(
                            discard_state, valid_discard_indices, "discard"
                        )
                        episode.discard_decisions.append(
                            (discard_state, card_idx, log_prob, entropy, current_pos)
                        )
                        episode.discard_rewards.append(0.0)
                    else:
                        card_idx = 0
                else:
                    card_idx = (
                        random.choice(valid_discard_indices)
                        if valid_discard_indices
                        else 0
                    )

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

                    if is_model_position:
                        if valid_card_indices:
                            card_idx, log_prob, entropy = self.select_action_with_exploration(
                                state_encoding, valid_card_indices, "card"
                            )
                            episode.card_decisions.append(
                                (state_encoding, card_idx, log_prob, entropy, current_pos)
                            )
                            # Card decisions start with 0 reward; trick outcome
                            # is assigned immediately when the trick completes
                            episode.card_rewards.append(0.0)
                        else:
                            card_idx = 0
                    else:
                        card_idx = (
                            random.choice(valid_card_indices)
                            if valid_card_indices
                            else 0
                        )

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

                    # TRICK-LEVEL REWARDS: assign to card decisions in this trick
                    if result.get("trick_complete"):
                        winner_pos = result["trick_winner"]
                        current_hand_info["tricks_won_by_pos"][winner_pos] += 1

                        if winner_pos in [0, 2]:  # Model's team won trick
                            trick_reward = 0.05
                        else:
                            trick_reward = -0.02

                        # Assign trick reward to all model card decisions
                        # in this trick (most recent decisions since last trick)
                        for i in range(len(episode.card_rewards) - 1, hand_card_start_idx - 1, -1):
                            if i >= 0 and i < len(episode.card_rewards):
                                # Only assign to decisions that don't have
                                # a trick reward yet (still at 0.0 or negative small)
                                episode.card_rewards[i] += trick_reward

                    # HAND-LEVEL REWARDS
                    if result.get("hand_complete"):
                        hand_winner = result["hand_winner"]
                        winning_team = hand_winner["winning_team"]
                        points_awarded = hand_winner["points_awarded"]
                        calling_team = current_hand_info["calling_team"]

                        # Balanced rewards: calling is not punished more than
                        # it is rewarded. The key insight is that in Euchre,
                        # you MUST call sometimes (dealer is forced), so the
                        # reward for a successful call should be >= the penalty
                        # for being euchred, weighted by probability.
                        if calling_team == 1:  # Model's team called
                            if winning_team == 1:
                                if points_awarded == 2:
                                    hand_reward = 0.40  # March - big reward
                                else:
                                    hand_reward = 0.20  # Made it
                            else:
                                hand_reward = -0.25  # Euchred (reduced from -0.30)
                                episode.euchres[0] += 1
                                episode.euchres[2] += 1
                        elif calling_team == 2:  # Opponent called
                            if winning_team == 1:
                                hand_reward = 0.20  # Euchred opponent (reduced from 0.30)
                            else:
                                hand_reward = -0.10  # Opponent made it
                        else:
                            hand_reward = 0.0

                        # Assign hand reward to all model decisions in this hand
                        for i in range(hand_trump_start_idx, len(episode.trump_rewards)):
                            episode.trump_rewards[i] += hand_reward
                        for i in range(hand_card_start_idx, len(episode.card_rewards)):
                            episode.card_rewards[i] += hand_reward
                        for i in range(hand_discard_start_idx, len(episode.discard_rewards)):
                            episode.discard_rewards[i] += hand_reward

                        # Update hand boundary indices for next hand
                        hand_card_start_idx = len(episode.card_rewards)
                        hand_trump_start_idx = len(episode.trump_rewards)
                        hand_discard_start_idx = len(episode.discard_rewards)

                        # Reset hand info
                        current_hand_info = {
                            "caller_position": None,
                            "calling_team": None,
                            "tricks_won_by_pos": {0: 0, 1: 0, 2: 0, 3: 0},
                            "was_forced_call": False,
                        }

                        # Start new hand if game not over
                        if game.state.phase != GamePhase.GAME_OVER:
                            game.start_new_hand()

            elif game.state.phase == GamePhase.HAND_COMPLETE:
                game.start_new_hand()

        # GAME-LEVEL REWARD (score differential, applied to all decisions)
        episode.team1_score = game.state.team1_score
        episode.team2_score = game.state.team2_score
        episode.game_reward = (game.state.team1_score - game.state.team2_score) / 10.0

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

        Key fixes over the original:
        1. Per-decision returns with proper temporal credit assignment
        2. Actual entropy regularization (not dead code)
        3. Per-decision-type baselines for better variance reduction
        4. Advantage normalization to stabilize training
        """
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
        total_wins = 0

        for episode in episodes:
            game_reward = episode.game_reward
            total_reward += game_reward

            if game_reward > 0:
                total_wins += 1

            # Card decisions: each has its own reward (trick + hand outcome)
            # plus discounted game reward. Compute returns from per-decision
            # reward sequence so later decisions get less game-level discount.
            card_reward_seq = [r + game_reward for r in episode.card_rewards]
            card_ret = self.compute_returns(card_reward_seq, self.gamma)

            for i, (state, action, log_prob, entropy, pos) in enumerate(episode.card_decisions):
                card_log_probs.append(log_prob)
                card_entropies.append(entropy)
                card_returns.append(card_ret[i] if i < len(card_ret) else game_reward)

            # Trump decisions: reward is hand outcome + game reward
            trump_reward_seq = [r + game_reward for r in episode.trump_rewards]
            trump_ret = self.compute_returns(trump_reward_seq, self.gamma)

            for i, (state, action, log_prob, entropy, pos) in enumerate(episode.trump_decisions):
                trump_log_probs.append(log_prob)
                trump_entropies.append(entropy)
                trump_returns.append(trump_ret[i] if i < len(trump_ret) else game_reward)

            # Discard decisions
            discard_reward_seq = [r + game_reward for r in episode.discard_rewards]
            discard_ret = self.compute_returns(discard_reward_seq, self.gamma)

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

        if card_log_probs:
            card_lp = torch.stack([lp.squeeze() if lp.dim() > 0 else lp for lp in card_log_probs])
            card_ret_t = torch.FloatTensor(card_returns).to(self.device)
            # Normalize advantages per decision type for stable training
            card_advantages = card_ret_t - card_ret_t.mean()
            if card_ret_t.numel() > 1:
                std = card_ret_t.std()
                if std > 1e-8:
                    card_advantages = card_advantages / std

            card_loss = -(card_lp * card_advantages).mean()
            total_loss = total_loss + card_loss

            card_ent = torch.stack([e.squeeze() if e.dim() > 0 else e for e in card_entropies])
            total_entropy = total_entropy + card_ent.mean()

        if trump_log_probs:
            trump_lp = torch.stack([lp.squeeze() if lp.dim() > 0 else lp for lp in trump_log_probs])
            trump_ret_t = torch.FloatTensor(trump_returns).to(self.device)
            trump_advantages = trump_ret_t - trump_ret_t.mean()
            if trump_ret_t.numel() > 1:
                std = trump_ret_t.std()
                if std > 1e-8:
                    trump_advantages = trump_advantages / std

            trump_loss = -(trump_lp * trump_advantages).mean()
            total_loss = total_loss + trump_loss

            trump_ent = torch.stack([e.squeeze() if e.dim() > 0 else e for e in trump_entropies])
            total_entropy = total_entropy + trump_ent.mean()

        if discard_log_probs:
            discard_lp = torch.stack([lp.squeeze() if lp.dim() > 0 else lp for lp in discard_log_probs])
            discard_ret_t = torch.FloatTensor(discard_returns).to(self.device)
            discard_advantages = discard_ret_t - discard_ret_t.mean()
            if discard_ret_t.numel() > 1:
                std = discard_ret_t.std()
                if std > 1e-8:
                    discard_advantages = discard_advantages / std

            discard_loss = -(discard_lp * discard_advantages).mean()
            total_loss = total_loss + discard_loss

            discard_ent = torch.stack([e.squeeze() if e.dim() > 0 else e for e in discard_entropies])
            total_entropy = total_entropy + discard_ent.mean()

        # Entropy bonus: encourages exploration, prevents policy collapse
        total_loss = total_loss - self.entropy_beta * total_entropy

        # Backpropagation
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        # Update statistics
        self.total_games += len(episodes)
        self.total_wins += total_wins
        self.avg_reward = total_reward / len(episodes)

        return {
            "loss": total_loss.item(),
            "avg_reward": self.avg_reward,
            "win_rate": total_wins / len(episodes),
            "running_reward": self.running_reward,
            "entropy": total_entropy.item(),
        }
