"""
PPO (Proximal Policy Optimization) Trainer for Euchre AI
Uses Actor-Critic with GAE, clipped surrogate objective, and optional self-play
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
from collections import defaultdict, namedtuple

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
from networks.critic_nn import EuchreCritic


# Stores a single decision step with all info needed for PPO updates
Transition = namedtuple(
    "Transition",
    ["state", "action_idx", "log_prob", "value", "reward", "decision_type", "position"],
)


class Episode:
    """Records a single game episode with all transitions and game stats"""

    def __init__(self):
        # Store transitions by decision type
        self.card_transitions = []
        self.trump_transitions = []
        self.discard_transitions = []

        # Game-level reward applied to all decisions as a final bonus
        self.game_reward = 0.0

        # Track game statistics
        self.team1_score = 0
        self.team2_score = 0
        self.trump_calls = defaultdict(int)
        self.euchres = defaultdict(int)


class PPOTrainer:
    """Trains actor-critic models using Proximal Policy Optimization"""

    def __init__(
        self,
        model: BasicEuchreNN,
        critic: EuchreCritic,
        lr_actor: float = 0.0003,
        lr_critic: float = 0.001,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_beta: float = 0.02,
        ppo_epochs: int = 4,
        mini_batch_size: int = 64,
        exploration_rate: float = 0.1,
        use_cuda: bool = True,
        self_play: bool = False,
        opponent_update_interval: int = 20,
    ):
        self.model = model
        self.critic = critic
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_beta = entropy_beta
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size
        self.exploration_rate = exploration_rate
        self.self_play = self_play
        self.opponent_update_interval = opponent_update_interval

        # Determine device
        self.device = torch.device(
            "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)
        self.critic.to(self.device)

        # Separate optimizers for actor and critic
        self.actor_optimizer = optim.Adam(self.model.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        # Opponent model for self-play (frozen copy)
        self.opponent_model = None
        # Pool of past opponent snapshots for league training
        self.opponent_pool = []

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

    def select_action(
        self,
        state_encoding: np.ndarray,
        valid_indices: List[int],
        decision_type: str,
    ) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Select action using model's policy with epsilon-greedy exploration.
        Also gets value estimate from critic.

        Returns:
            (action_idx, log_prob, value, entropy)
        """
        state_tensor = torch.FloatTensor(state_encoding).unsqueeze(0).to(self.device)

        # Get logits from actor network
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

        # Create distribution from masked logits
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

        # Get value estimate from critic
        with torch.no_grad():
            if decision_type == "card":
                value = self.critic.forward_card(state_tensor)
            elif decision_type == "trump":
                value = self.critic.forward_trump(state_tensor)
            else:  # discard
                value = self.critic.forward_discard(state_tensor)

        return action_idx, log_prob, value.squeeze(), entropy

    def _opponent_select_action(
        self,
        state_encoding: np.ndarray,
        valid_indices: List[int],
        decision_type: str,
    ) -> int:
        """
        Select action for opponent using opponent model or random play.

        Returns:
            action_idx
        """
        if self.opponent_model is None:
            return random.choice(valid_indices)

        state_tensor = torch.FloatTensor(state_encoding).unsqueeze(0).to(self.device)

        with torch.no_grad():
            if decision_type == "card":
                logits = self.opponent_model.forward(state_tensor, return_logits=True)
            elif decision_type == "trump":
                logits = self.opponent_model.forward_trump(state_tensor, return_logits=True)
            else:  # discard
                logits = self.opponent_model.forward_discard(state_tensor, return_logits=True)

            # Mask invalid actions
            mask = torch.full_like(logits, float("-inf"))
            mask[0, valid_indices] = 0
            masked_logits = logits + mask

            dist = torch.distributions.Categorical(logits=masked_logits)
            action_tensor = dist.sample()

        return action_tensor.item()

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

    def _select_opponent(self):
        """
        Select an opponent from the pool for league-style training.
        70% chance: use latest opponent, 30% chance: random from pool.
        """
        if not self.opponent_pool:
            return None

        if random.random() < 0.7:
            return self.opponent_pool[-1]
        else:
            return random.choice(self.opponent_pool)

    def play_game(self) -> Episode:
        """
        Play a single game with the model as partners (positions 0 & 2)
        and opponent (self-play or random) as opponents (positions 1 & 3).

        Per-decision rewards are tracked so each decision gets credit for
        outcomes it directly influenced.
        """
        episode = Episode()

        # Select opponent for this game if using self-play
        if self.self_play:
            selected_opponent = self._select_opponent()
            if selected_opponent is not None:
                self.opponent_model = selected_opponent

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

        # Track which transitions belong to the current hand so we can
        # assign hand-outcome rewards to only those transitions
        hand_card_start_idx = 0
        hand_trump_start_idx = 0
        hand_discard_start_idx = 0

        # Start game
        game.start_new_hand()

        # Play until game over
        while game.state.phase != GamePhase.GAME_OVER:
            current_pos = game.state.current_player_position
            is_model_position = current_pos in [0, 2]
            is_opponent_position = current_pos in [1, 3]

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
                    decision_idx, log_prob, value, entropy = self.select_action(
                        trump_state, valid_trump_indices, "trump"
                    )
                    transition = Transition(
                        state=trump_state,
                        action_idx=decision_idx,
                        log_prob=log_prob,
                        value=value,
                        reward=0.0,
                        decision_type="trump",
                        position=current_pos,
                    )
                    episode.trump_transitions.append(transition)
                elif is_opponent_position:
                    decision_idx = self._opponent_select_action(
                        trump_state, valid_trump_indices, "trump"
                    )
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

                                # REWARD SHAPING: hand strength bonus when model calls trump
                                suit_char_map = {
                                    Suit.CLUBS: "C",
                                    Suit.DIAMONDS: "D",
                                    Suit.HEARTS: "H",
                                    Suit.SPADES: "S",
                                }
                                trump_suit_char = suit_char_map.get(
                                    game.state.turned_up_card.suit, "C"
                                )
                                player = game.state.get_player(current_pos)
                                hand_cards = [str(card) for card in player.hand]

                                # Compute hand strength
                                trump_count = sum(
                                    1 for c in hand_cards if c.endswith(trump_suit_char)
                                ) / 5.0
                                has_right_bower = 1.0 if f"J{trump_suit_char}" in hand_cards else 0.0
                                color_map = {"C": "S", "S": "C", "D": "H", "H": "D"}
                                same_color_suit = color_map.get(trump_suit_char, "C")
                                has_left_bower = 1.0 if f"J{same_color_suit}" in hand_cards else 0.0
                                off_aces = sum(
                                    1
                                    for c in hand_cards
                                    if c.startswith("A") and not c.endswith(trump_suit_char)
                                ) / 3.0
                                strength = (
                                    trump_count * 2
                                    + has_right_bower * 3
                                    + has_left_bower * 2.5
                                    + off_aces * 1.5
                                ) / 10.0
                                call_bonus = strength * 0.03

                                # Add call_bonus to the most recent trump transition
                                if episode.trump_transitions:
                                    old_t = episode.trump_transitions[-1]
                                    episode.trump_transitions[-1] = Transition(
                                        state=old_t.state,
                                        action_idx=old_t.action_idx,
                                        log_prob=old_t.log_prob,
                                        value=old_t.value,
                                        reward=old_t.reward + call_bonus,
                                        decision_type=old_t.decision_type,
                                        position=old_t.position,
                                    )
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
                    decision_idx, log_prob, value, entropy = self.select_action(
                        trump_state, valid_trump_indices, "trump"
                    )
                    transition = Transition(
                        state=trump_state,
                        action_idx=decision_idx,
                        log_prob=log_prob,
                        value=value,
                        reward=0.0,
                        decision_type="trump",
                        position=current_pos,
                    )
                    episode.trump_transitions.append(transition)
                elif is_opponent_position:
                    decision_idx = self._opponent_select_action(
                        trump_state, valid_trump_indices, "trump"
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

                            # REWARD SHAPING: hand strength bonus when model calls in round 2
                            suit_char_map = {
                                Suit.CLUBS: "C",
                                Suit.DIAMONDS: "D",
                                Suit.HEARTS: "H",
                                Suit.SPADES: "S",
                            }
                            trump_suit_char = suit_char_map.get(selected_suit, "C")
                            player = game.state.get_player(current_pos)
                            hand_cards = [str(card) for card in player.hand]

                            trump_count = sum(
                                1 for c in hand_cards if c.endswith(trump_suit_char)
                            ) / 5.0
                            has_right_bower = 1.0 if f"J{trump_suit_char}" in hand_cards else 0.0
                            color_map = {"C": "S", "S": "C", "D": "H", "H": "D"}
                            same_color_suit = color_map.get(trump_suit_char, "C")
                            has_left_bower = 1.0 if f"J{same_color_suit}" in hand_cards else 0.0
                            off_aces = sum(
                                1
                                for c in hand_cards
                                if c.startswith("A") and not c.endswith(trump_suit_char)
                            ) / 3.0
                            strength = (
                                trump_count * 2
                                + has_right_bower * 3
                                + has_left_bower * 2.5
                                + off_aces * 1.5
                            ) / 10.0
                            call_bonus = strength * 0.03

                            if episode.trump_transitions:
                                old_t = episode.trump_transitions[-1]
                                episode.trump_transitions[-1] = Transition(
                                    state=old_t.state,
                                    action_idx=old_t.action_idx,
                                    log_prob=old_t.log_prob,
                                    value=old_t.value,
                                    reward=old_t.reward + call_bonus,
                                    decision_type=old_t.decision_type,
                                    position=old_t.position,
                                )
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
                        card_idx, log_prob, value, entropy = self.select_action(
                            discard_state, valid_discard_indices, "discard"
                        )
                        transition = Transition(
                            state=discard_state,
                            action_idx=card_idx,
                            log_prob=log_prob,
                            value=value,
                            reward=0.0,
                            decision_type="discard",
                            position=current_pos,
                        )
                        episode.discard_transitions.append(transition)
                    else:
                        card_idx = 0
                elif is_opponent_position:
                    card_idx = (
                        self._opponent_select_action(
                            discard_state, valid_discard_indices, "discard"
                        )
                        if valid_discard_indices
                        else 0
                    )
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
                            card_idx, log_prob, value, entropy = self.select_action(
                                state_encoding, valid_card_indices, "card"
                            )
                            transition = Transition(
                                state=state_encoding,
                                action_idx=card_idx,
                                log_prob=log_prob,
                                value=value,
                                reward=0.0,
                                decision_type="card",
                                position=current_pos,
                            )
                            episode.card_transitions.append(transition)
                        else:
                            card_idx = 0
                    elif is_opponent_position:
                        card_idx = (
                            self._opponent_select_action(
                                state_encoding, valid_card_indices, "card"
                            )
                            if valid_card_indices
                            else 0
                        )
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

                    # TRICK-LEVEL REWARDS: assign to card transitions in this trick
                    if result.get("trick_complete"):
                        winner_pos = result["trick_winner"]
                        current_hand_info["tricks_won_by_pos"][winner_pos] += 1

                        if winner_pos in [0, 2]:  # Model's team won trick
                            trick_reward = 0.05
                        else:
                            trick_reward = -0.02

                        # Assign trick reward to all model card transitions in this trick
                        for i in range(
                            len(episode.card_transitions) - 1,
                            hand_card_start_idx - 1,
                            -1,
                        ):
                            if 0 <= i < len(episode.card_transitions):
                                old_t = episode.card_transitions[i]
                                episode.card_transitions[i] = Transition(
                                    state=old_t.state,
                                    action_idx=old_t.action_idx,
                                    log_prob=old_t.log_prob,
                                    value=old_t.value,
                                    reward=old_t.reward + trick_reward,
                                    decision_type=old_t.decision_type,
                                    position=old_t.position,
                                )

                        # PARTNER COORDINATION: when partner wins a trick, bonus
                        # to recent card decisions
                        partner_positions = {0: 2, 2: 0}
                        for model_pos in [0, 2]:
                            partner_pos = partner_positions[model_pos]
                            if winner_pos == partner_pos:
                                # Add small bonus to recent card transitions from model_pos
                                for i in range(
                                    len(episode.card_transitions) - 1,
                                    hand_card_start_idx - 1,
                                    -1,
                                ):
                                    if 0 <= i < len(episode.card_transitions):
                                        t = episode.card_transitions[i]
                                        if t.position == model_pos:
                                            episode.card_transitions[i] = Transition(
                                                state=t.state,
                                                action_idx=t.action_idx,
                                                log_prob=t.log_prob,
                                                value=t.value,
                                                reward=t.reward + 0.025,
                                                decision_type=t.decision_type,
                                                position=t.position,
                                            )

                    # HAND-LEVEL REWARDS
                    if result.get("hand_complete"):
                        hand_winner = result["hand_winner"]
                        winning_team = hand_winner["winning_team"]
                        points_awarded = hand_winner["points_awarded"]
                        calling_team = current_hand_info["calling_team"]

                        if calling_team == 1:  # Model's team called
                            if winning_team == 1:
                                if points_awarded == 2:
                                    hand_reward = 0.40  # March
                                else:
                                    hand_reward = 0.20  # Made it
                            else:
                                hand_reward = -0.25  # Euchred
                                episode.euchres[0] += 1
                                episode.euchres[2] += 1
                        elif calling_team == 2:  # Opponent called
                            if winning_team == 1:
                                hand_reward = 0.20  # Euchred opponent
                            else:
                                hand_reward = -0.10  # Opponent made it
                        else:
                            hand_reward = 0.0

                        # Assign hand reward to all model transitions in this hand
                        for i in range(
                            hand_trump_start_idx, len(episode.trump_transitions)
                        ):
                            old_t = episode.trump_transitions[i]
                            episode.trump_transitions[i] = Transition(
                                state=old_t.state,
                                action_idx=old_t.action_idx,
                                log_prob=old_t.log_prob,
                                value=old_t.value,
                                reward=old_t.reward + hand_reward,
                                decision_type=old_t.decision_type,
                                position=old_t.position,
                            )
                        for i in range(
                            hand_card_start_idx, len(episode.card_transitions)
                        ):
                            old_t = episode.card_transitions[i]
                            episode.card_transitions[i] = Transition(
                                state=old_t.state,
                                action_idx=old_t.action_idx,
                                log_prob=old_t.log_prob,
                                value=old_t.value,
                                reward=old_t.reward + hand_reward,
                                decision_type=old_t.decision_type,
                                position=old_t.position,
                            )
                        for i in range(
                            hand_discard_start_idx, len(episode.discard_transitions)
                        ):
                            old_t = episode.discard_transitions[i]
                            episode.discard_transitions[i] = Transition(
                                state=old_t.state,
                                action_idx=old_t.action_idx,
                                log_prob=old_t.log_prob,
                                value=old_t.value,
                                reward=old_t.reward + hand_reward,
                                decision_type=old_t.decision_type,
                                position=old_t.position,
                            )

                        # Update hand boundary indices for next hand
                        hand_card_start_idx = len(episode.card_transitions)
                        hand_trump_start_idx = len(episode.trump_transitions)
                        hand_discard_start_idx = len(episode.discard_transitions)

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

        # GAME-LEVEL REWARD (score differential, applied to all transitions)
        episode.team1_score = game.state.team1_score
        episode.team2_score = game.state.team2_score
        episode.game_reward = (game.state.team1_score - game.state.team2_score) / 10.0

        return episode

    def compute_gae(
        self,
        rewards: List[float],
        values: List[float],
        gamma: float,
        gae_lambda: float,
    ) -> Tuple[List[float], List[float]]:
        """
        Compute Generalized Advantage Estimation (GAE).

        Args:
            rewards: List of per-step rewards
            values: List of per-step value estimates V(s)
            gamma: Discount factor
            gae_lambda: GAE lambda for bias-variance tradeoff

        Returns:
            (returns, advantages) as lists of floats
        """
        # Append 0 for bootstrap value of terminal state
        values_extended = list(values) + [0.0]

        # Compute TD residuals (deltas)
        deltas = []
        for t in range(len(rewards)):
            delta = rewards[t] + gamma * values_extended[t + 1] - values_extended[t]
            deltas.append(delta)

        # Compute advantages using exponential moving average
        advantages = []
        gae = 0.0
        for t in reversed(range(len(deltas))):
            gae = deltas[t] + gamma * gae_lambda * gae
            advantages.insert(0, gae)

        # Returns = advantages + values
        returns = [adv + val for adv, val in zip(advantages, values)]

        return returns, advantages

    def train_on_batch(self, episodes: List[Episode]) -> Dict[str, float]:
        """
        Train actor and critic on a batch of episodes using PPO.

        Collects all transitions, computes GAE, then runs multiple epochs
        of clipped surrogate optimization.
        """
        # Collect all transitions by type with game reward added
        card_data = []  # (state, action, old_log_prob, value, reward)
        trump_data = []
        discard_data = []

        total_reward = 0.0
        total_wins = 0

        for episode in episodes:
            game_reward = episode.game_reward
            total_reward += game_reward
            if game_reward > 0:
                total_wins += 1

            for t in episode.card_transitions:
                card_data.append((
                    t.state,
                    t.action_idx,
                    t.log_prob.detach(),
                    t.value.detach().item() if isinstance(t.value, torch.Tensor) else t.value,
                    t.reward + game_reward,
                ))

            for t in episode.trump_transitions:
                trump_data.append((
                    t.state,
                    t.action_idx,
                    t.log_prob.detach(),
                    t.value.detach().item() if isinstance(t.value, torch.Tensor) else t.value,
                    t.reward + game_reward,
                ))

            for t in episode.discard_transitions:
                discard_data.append((
                    t.state,
                    t.action_idx,
                    t.log_prob.detach(),
                    t.value.detach().item() if isinstance(t.value, torch.Tensor) else t.value,
                    t.reward + game_reward,
                ))

        # Update running reward baseline
        if self.running_reward is None:
            self.running_reward = total_reward / len(episodes)
        else:
            self.running_reward = 0.95 * self.running_reward + 0.05 * (
                total_reward / len(episodes)
            )

        # Compute GAE returns and advantages for each decision type
        def prepare_gae(data_list):
            if not data_list:
                return [], [], [], [], []
            states, actions, old_log_probs, values, rewards = zip(*data_list)
            returns, advantages = self.compute_gae(
                list(rewards), list(values), self.gamma, self.gae_lambda
            )
            # Normalize advantages
            adv_tensor = torch.FloatTensor(advantages)
            if adv_tensor.numel() > 1:
                std = adv_tensor.std()
                if std > 1e-8:
                    adv_tensor = (adv_tensor - adv_tensor.mean()) / std
                else:
                    adv_tensor = adv_tensor - adv_tensor.mean()
            advantages_normalized = adv_tensor.tolist()
            return (
                list(states),
                list(actions),
                list(old_log_probs),
                list(returns),
                advantages_normalized,
            )

        card_states, card_actions, card_old_lps, card_returns, card_advs = prepare_gae(card_data)
        trump_states, trump_actions, trump_old_lps, trump_returns, trump_advs = prepare_gae(trump_data)
        discard_states, discard_actions, discard_old_lps, discard_returns, discard_advs = prepare_gae(discard_data)

        # Track stats across PPO epochs
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy_val = 0.0
        total_clip_fraction = 0.0
        num_updates = 0

        # Run multiple PPO epochs
        for epoch in range(self.ppo_epochs):
            epoch_policy_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            epoch_value_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            epoch_entropy = torch.tensor(0.0, device=self.device)
            epoch_clip_count = 0
            epoch_total_count = 0

            # Process each decision type
            for dtype, states, actions, old_lps, returns, advs in [
                ("card", card_states, card_actions, card_old_lps, card_returns, card_advs),
                ("trump", trump_states, trump_actions, trump_old_lps, trump_returns, trump_advs),
                ("discard", discard_states, discard_actions, discard_old_lps, discard_returns, discard_advs),
            ]:
                if not states:
                    continue

                n = len(states)

                # Convert to tensors
                state_tensor = torch.FloatTensor(np.array(states)).to(self.device)
                action_tensor = torch.LongTensor(actions).to(self.device)
                old_lp_tensor = torch.stack(old_lps).to(self.device)
                returns_tensor = torch.FloatTensor(returns).to(self.device)
                advs_tensor = torch.FloatTensor(advs).to(self.device)

                # Get new logits from current model
                if dtype == "card":
                    new_logits = self.model.forward(state_tensor, return_logits=True)
                elif dtype == "trump":
                    new_logits = self.model.forward_trump(state_tensor, return_logits=True)
                else:
                    new_logits = self.model.forward_discard(state_tensor, return_logits=True)

                # We don't re-mask here because the old actions are already valid;
                # we just need log_prob of those specific actions under new policy.
                # Use logits directly - invalid actions will have very low probability
                # but the actions we took were always valid.
                new_dist = torch.distributions.Categorical(logits=new_logits)
                new_log_probs = new_dist.log_prob(action_tensor)
                entropy = new_dist.entropy()

                # Compute ratio
                ratio = torch.exp(new_log_probs - old_lp_tensor)

                # Clipped surrogate objective
                surr1 = ratio * advs_tensor
                surr2 = (
                    torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon)
                    * advs_tensor
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                # Get new value estimates from critic
                if dtype == "card":
                    new_values = self.critic.forward_card(state_tensor).squeeze()
                elif dtype == "trump":
                    new_values = self.critic.forward_trump(state_tensor).squeeze()
                else:
                    new_values = self.critic.forward_discard(state_tensor).squeeze()

                # Value loss (MSE)
                value_loss = nn.functional.mse_loss(new_values, returns_tensor)

                # Entropy loss
                entropy_loss = -entropy.mean()

                # Accumulate losses
                epoch_policy_loss = epoch_policy_loss + policy_loss
                epoch_value_loss = epoch_value_loss + value_loss
                epoch_entropy = epoch_entropy + entropy.mean()

                # Track clip fraction
                with torch.no_grad():
                    clip_fraction = (
                        (torch.abs(ratio - 1.0) > self.clip_epsilon).float().sum().item()
                    )
                    epoch_clip_count += clip_fraction
                    epoch_total_count += n

            # Combined loss across all types
            combined_loss = (
                epoch_policy_loss
                + 0.5 * epoch_value_loss
                + self.entropy_beta * (-epoch_entropy)  # entropy_beta * entropy_loss
            )

            # Backprop with gradient clipping
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            combined_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
            self.actor_optimizer.step()
            self.critic_optimizer.step()

            # Accumulate stats
            total_policy_loss += epoch_policy_loss.item()
            total_value_loss += epoch_value_loss.item()
            total_entropy_val += epoch_entropy.item()
            if epoch_total_count > 0:
                total_clip_fraction += epoch_clip_count / epoch_total_count
            num_updates += 1

        # Average stats over epochs
        avg_policy_loss = total_policy_loss / max(1, num_updates)
        avg_value_loss = total_value_loss / max(1, num_updates)
        avg_entropy = total_entropy_val / max(1, num_updates)
        avg_clip_fraction = total_clip_fraction / max(1, num_updates)

        # Update statistics
        self.total_games += len(episodes)
        self.total_wins += total_wins
        self.avg_reward = total_reward / len(episodes)

        return {
            "loss": avg_policy_loss + 0.5 * avg_value_loss,
            "avg_reward": self.avg_reward,
            "win_rate": total_wins / len(episodes),
            "running_reward": self.running_reward,
            "entropy": avg_entropy,
            "policy_loss": avg_policy_loss,
            "value_loss": avg_value_loss,
            "clip_fraction": avg_clip_fraction,
        }
