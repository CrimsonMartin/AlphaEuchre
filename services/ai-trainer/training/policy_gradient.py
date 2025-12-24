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
        # Store decisions by type
        self.card_decisions = []  # (state, action_idx, log_prob, position)
        self.trump_decisions = []  # (state, action_idx, log_prob, position)
        self.discard_decisions = []  # (state, action_idx, log_prob, position)

        # Store rewards at different levels
        self.trick_rewards = defaultdict(list)  # position -> [rewards]
        self.hand_rewards = defaultdict(list)  # position -> [rewards]
        self.game_reward = 0.0  # Final game outcome

        # Track game statistics
        self.team1_score = 0
        self.team2_score = 0
        self.tricks_won = defaultdict(int)  # position -> count
        self.trump_calls = defaultdict(int)  # position -> count
        self.euchres = defaultdict(int)  # position -> count


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
    ) -> Tuple[int, torch.Tensor]:
        """
        Select action using model's policy with epsilon-greedy exploration.

        Args:
            state_encoding: Encoded game state
            valid_indices: List of valid action indices
            decision_type: 'card', 'trump', or 'discard'

        Returns:
            (action_index, log_probability)
        """
        # Epsilon-greedy exploration
        if random.random() < self.exploration_rate:
            # Random action
            action_idx = random.choice(valid_indices)

            # Still need to compute log prob for gradient
            state_tensor = (
                torch.FloatTensor(state_encoding).unsqueeze(0).to(self.device)
            )

            with torch.no_grad():
                if decision_type == "card":
                    probs = self.model.forward(state_tensor)
                elif decision_type == "trump":
                    probs = self.model.forward_trump(state_tensor)
                else:  # discard
                    probs = self.model.forward_discard(state_tensor)

                # Mask invalid actions
                mask = torch.full_like(probs, float("-inf"))
                mask[0, valid_indices] = 0
                masked_probs = probs + mask
                masked_probs = torch.softmax(masked_probs, dim=1)

                log_prob = torch.log(masked_probs[0, action_idx] + 1e-10)

            return action_idx, log_prob

        # Use model's policy
        state_tensor = torch.FloatTensor(state_encoding).unsqueeze(0).to(self.device)

        if decision_type == "card":
            probs = self.model.forward(state_tensor)
        elif decision_type == "trump":
            probs = self.model.forward_trump(state_tensor)
        else:  # discard
            probs = self.model.forward_discard(state_tensor)

        # Mask invalid actions
        mask = torch.full_like(probs, float("-inf"))
        mask[0, valid_indices] = 0
        masked_probs = probs + mask
        masked_probs = torch.softmax(masked_probs, dim=1)

        # Sample from distribution
        dist = torch.distributions.Categorical(masked_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.item(), log_prob

    def play_game(self) -> Episode:
        """
        Play a single game with the model as partners (positions 0 & 2)
        and random AI as opponents (positions 1 & 3).

        Returns:
            Episode with recorded decisions and rewards
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
        }

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

                # Determine valid options
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
                    valid_trump_indices = [turned_up_suit_idx, 4]  # Call or pass
                else:
                    valid_trump_indices = [4]  # Only pass

                if is_model_position:
                    # Model makes decision
                    decision_idx, log_prob = self.select_action_with_exploration(
                        trump_state, valid_trump_indices, "trump"
                    )
                    episode.trump_decisions.append(
                        (trump_state, decision_idx, log_prob, current_pos)
                    )
                else:
                    # Random AI
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
                    decision_idx, log_prob = self.select_action_with_exploration(
                        trump_state, valid_trump_indices, "trump"
                    )
                    episode.trump_decisions.append(
                        (trump_state, decision_idx, log_prob, current_pos)
                    )
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
                        card_idx, log_prob = self.select_action_with_exploration(
                            discard_state, valid_discard_indices, "discard"
                        )
                        episode.discard_decisions.append(
                            (discard_state, card_idx, log_prob, current_pos)
                        )
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
                            card_idx, log_prob = self.select_action_with_exploration(
                                state_encoding, valid_card_indices, "card"
                            )
                            episode.card_decisions.append(
                                (state_encoding, card_idx, log_prob, current_pos)
                            )
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

                    # TRICK-LEVEL REWARDS
                    if result.get("trick_complete"):
                        winner_pos = result["trick_winner"]
                        current_hand_info["tricks_won_by_pos"][winner_pos] += 1

                        # Reward for winning trick
                        if winner_pos in [0, 2]:  # Model's team
                            trick_reward = 0.02
                            # Bonus if model's team called trump
                            if current_hand_info["calling_team"] == 1:
                                trick_reward += 0.01

                            episode.trick_rewards[0].append(trick_reward)
                            episode.trick_rewards[2].append(trick_reward)

                    # Check if hand complete
                    if result.get("hand_complete"):
                        # HAND-LEVEL REWARDS
                        hand_winner = result["hand_winner"]
                        winning_team = hand_winner["winning_team"]
                        points_awarded = hand_winner["points_awarded"]
                        calling_team = current_hand_info["calling_team"]

                        if calling_team == 1:  # Model called
                            if winning_team == 1:
                                # Model's team won
                                if points_awarded == 2:
                                    hand_reward = 0.25  # March
                                else:
                                    hand_reward = 0.10  # Made it
                            else:
                                # Model got euchred
                                hand_reward = -0.30
                                episode.euchres[0] += 1
                                episode.euchres[2] += 1
                        elif calling_team == 2:  # Opponent called
                            if winning_team == 1:
                                # Model successfully defended
                                hand_reward = 0.30  # Euchred opponent
                            else:
                                # Opponent made it
                                hand_reward = -0.15
                        else:
                            hand_reward = 0.0

                        episode.hand_rewards[0].append(hand_reward)
                        episode.hand_rewards[2].append(hand_reward)

                        # Reset hand info
                        current_hand_info = {
                            "caller_position": None,
                            "calling_team": None,
                            "tricks_won_by_pos": {0: 0, 1: 0, 2: 0, 3: 0},
                        }

                        # Start new hand if game not over
                        if game.state.phase != GamePhase.GAME_OVER:
                            game.start_new_hand()

            elif game.state.phase == GamePhase.HAND_COMPLETE:
                game.start_new_hand()

        # GAME-LEVEL REWARD (score differential)
        episode.team1_score = game.state.team1_score
        episode.team2_score = game.state.team2_score
        episode.game_reward = (game.state.team1_score - game.state.team2_score) / 10.0

        # Track statistics
        episode.tricks_won[0] = sum(
            current_hand_info["tricks_won_by_pos"].get(0, 0) for _ in range(1)
        )
        episode.tricks_won[2] = sum(
            current_hand_info["tricks_won_by_pos"].get(2, 0) for _ in range(1)
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

        Args:
            episodes: List of Episode objects

        Returns:
            Dictionary with training statistics
        """
        # Collect all decisions and their returns
        card_states = []
        card_actions = []
        card_log_probs = []
        card_returns = []

        trump_states = []
        trump_actions = []
        trump_log_probs = []
        trump_returns = []

        discard_states = []
        discard_actions = []
        discard_log_probs = []
        discard_returns = []

        total_reward = 0.0
        total_wins = 0

        for episode in episodes:
            # Compute total reward for this episode
            game_reward = episode.game_reward
            total_reward += game_reward

            if game_reward > 0:
                total_wins += 1

            # Combine all rewards for model positions (0 and 2)
            all_rewards_pos0 = (
                list(episode.trick_rewards[0])
                + list(episode.hand_rewards[0])
                + [game_reward]
            )
            all_rewards_pos2 = (
                list(episode.trick_rewards[2])
                + list(episode.hand_rewards[2])
                + [game_reward]
            )

            # Process card decisions
            for state, action, log_prob, pos in episode.card_decisions:
                rewards = all_rewards_pos0 if pos == 0 else all_rewards_pos2
                returns = self.compute_returns(rewards, self.gamma)

                card_states.append(state)
                card_actions.append(action)
                card_log_probs.append(log_prob)
                card_returns.append(returns[0] if returns else 0.0)

            # Process trump decisions
            for state, action, log_prob, pos in episode.trump_decisions:
                rewards = all_rewards_pos0 if pos == 0 else all_rewards_pos2
                returns = self.compute_returns(rewards, self.gamma)

                trump_states.append(state)
                trump_actions.append(action)
                trump_log_probs.append(log_prob)
                trump_returns.append(returns[0] if returns else 0.0)

            # Process discard decisions
            for state, action, log_prob, pos in episode.discard_decisions:
                rewards = all_rewards_pos0 if pos == 0 else all_rewards_pos2
                returns = self.compute_returns(rewards, self.gamma)

                discard_states.append(state)
                discard_actions.append(action)
                discard_log_probs.append(log_prob)
                discard_returns.append(returns[0] if returns else 0.0)

        # Compute baseline (running average)
        if self.running_reward is None:
            self.running_reward = total_reward / len(episodes)
        else:
            self.running_reward = 0.95 * self.running_reward + 0.05 * (
                total_reward / len(episodes)
            )

        # Compute policy gradient loss
        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        entropy_loss = 0.0

        if card_log_probs:
            # Ensure all log_probs are scalars (0-dimensional tensors)
            card_log_probs_list = [
                lp.squeeze() if lp.dim() > 0 else lp for lp in card_log_probs
            ]
            card_log_probs_tensor = torch.stack(card_log_probs_list)
            card_returns_tensor = torch.FloatTensor(card_returns).to(self.device)
            card_returns_tensor = card_returns_tensor - self.running_reward  # Baseline

            card_loss = -(card_log_probs_tensor * card_returns_tensor).mean()
            total_loss = total_loss + card_loss

        if trump_log_probs:
            # Ensure all log_probs are scalars (0-dimensional tensors)
            trump_log_probs_list = [
                lp.squeeze() if lp.dim() > 0 else lp for lp in trump_log_probs
            ]
            trump_log_probs_tensor = torch.stack(trump_log_probs_list)
            trump_returns_tensor = torch.FloatTensor(trump_returns).to(self.device)
            trump_returns_tensor = trump_returns_tensor - self.running_reward

            trump_loss = -(trump_log_probs_tensor * trump_returns_tensor).mean()
            total_loss = total_loss + trump_loss

        if discard_log_probs:
            # Ensure all log_probs are scalars (0-dimensional tensors)
            discard_log_probs_list = [
                lp.squeeze() if lp.dim() > 0 else lp for lp in discard_log_probs
            ]
            discard_log_probs_tensor = torch.stack(discard_log_probs_list)
            discard_returns_tensor = torch.FloatTensor(discard_returns).to(self.device)
            discard_returns_tensor = discard_returns_tensor - self.running_reward

            discard_loss = -(discard_log_probs_tensor * discard_returns_tensor).mean()
            total_loss = total_loss + discard_loss

        # Add entropy bonus for exploration
        # (Entropy is already computed in the log_probs, so we approximate)
        total_loss = total_loss - self.entropy_beta * entropy_loss

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
        }
