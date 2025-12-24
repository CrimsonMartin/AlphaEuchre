"""
Genetic Algorithm for Evolving Euchre Neural Networks
Round-Robin Tournament System with Team-Based Evaluation
ELO Rating System with Parallel Processing
ENHANCED: Neural network-based trump selection and dealer discard
"""

import random
import torch
import copy
import sys
import os
from typing import List, Tuple, Dict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing

# Add parent directory to path to import euchre_core
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
from networks.architecture_registry import ArchitectureRegistry
from elo_system import ELORatingSystem


# Global flag for GPU usage in workers
USE_GPU_IN_WORKERS = os.getenv("USE_GPU_IN_WORKERS", "false").lower() == "true"


# Helper function for parallel processing (must be at module level)
def run_tournament_group_worker(args):
    """Worker function to run a tournament group in parallel"""
    models_state_dicts, games_per_pairing, all_cards, group_indices, use_gpu = args

    # Reconstruct models from state dicts
    # Use GPU if available and enabled, otherwise CPU
    use_cuda = use_gpu and torch.cuda.is_available()
    models = []
    for state_dict in models_state_dicts:
        model = BasicEuchreNN(use_cuda=use_cuda)
        model.load_state_dict(state_dict)
        model.eval()
        models.append(model)

    # Run tournament
    ga_temp = GeneticAlgorithm(games_per_pairing=games_per_pairing, use_cuda=use_cuda)
    ga_temp.all_cards = all_cards

    # Initialize ELO system for this group
    for idx in group_indices:
        ga_temp.elo_system.ratings[idx] = ga_temp.elo_system.initial_rating

    # Run round-robin tournament
    ga_temp.run_round_robin_tournament_elo(models, group_indices)

    # Return the ELO ratings for this group
    return {idx: ga_temp.elo_system.get_rating(idx) for idx in group_indices}


class GeneticAlgorithm:
    """Genetic algorithm for evolving neural network weights with round-robin tournaments and ELO ratings"""

    def __init__(
        self,
        population_size=32,
        mutation_rate=0.10,  # Small mutations for incremental progress
        mutation_strength=0.20,  # Strength of weight perturbations (increased from 0.05)
        crossover_rate=0.7,  # Keep for compatibility
        elite_size=1,  # Single champion for champion_variations strategy
        hands_per_model=50,  # Hands each model plays in sampled round-robin
        games_per_pairing=50,  # Alias for hands_per_model for compatibility
        num_workers=4,
        use_elo=True,
        use_cuda=None,
        parallel_mode="thread",  # "thread", "process", or "sequential"
        evolution_strategy="champion_variations",  # "champion_variations" or "traditional"
    ):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.base_mutation_rate = mutation_rate  # Store base for adaptive mutation
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
        self.hands_per_model = hands_per_model
        self.games_per_pairing = games_per_pairing  # Alias for compatibility
        self.num_workers = num_workers or max(1, multiprocessing.cpu_count())
        self.use_elo = use_elo
        self.parallel_mode = parallel_mode
        self.evolution_strategy = evolution_strategy
        self.population: List[BasicEuchreNN] = []
        self.elo_ratings: List[float] = []

        # GPU/CUDA settings
        if use_cuda is None:
            self.use_cuda = torch.cuda.is_available()
        else:
            self.use_cuda = use_cuda and torch.cuda.is_available()

        # ELO rating system
        self.elo_system = ELORatingSystem(initial_rating=1500, k_factor=32)

        # Global champion tracking (best model across all generations)
        self.global_best_model: BasicEuchreNN | None = None
        self.global_best_elo: float = 0.0

        # Stagnation tracking for adaptive mutation
        self.generations_without_improvement = 0
        self.stagnation_threshold = 5

        # Simulated Annealing parameters
        self.temperature = 1.0  # Initial temperature
        self.initial_temperature = 1.0
        self.max_temperature = 1.5
        self.cooling_rate = 0.95  # Adaptive cooling
        self.heating_rate = 1.2  # Reheat on stagnation

        # Architecture diversity tracking
        self.architecture_enabled = True  # Enable architecture support (uses transformer-only for best performance)
        self.architecture_stats = {}  # Track performance by architecture

        # Multi-objective fitness: Track per-model behavioral statistics
        self.model_stats: Dict[int, Dict] = {}  # model_index -> stats dict

        # Reward shaping parameters
        self.reward_shaping_enabled = True
        self.aggressive_win_bonus = (
            0.20  # +20% ELO bonus for winning when your team called
        )
        self.passive_win_penalty = (
            -0.15
        )  # -15% penalty for winning when opponent called
        self.euchre_bonus = 0.25  # +25% bonus for euchre (opponent called and lost)
        self.forced_call_penalty = -0.10  # -10% penalty for forced dealer call

        # Card mapping for predictions
        self.all_cards = []
        for suit in ["C", "D", "H", "S"]:
            for rank in ["9", "10", "J", "Q", "K", "A"]:
                self.all_cards.append(f"{rank}{suit}")

    def initialize_population(self, seed_models: List[BasicEuchreNN] | None = None):
        """
        Create initial population, optionally seeded with existing models.
        Supports multi-architecture populations.

        Args:
            seed_models: Optional list of models to seed the population
        """
        self.population = []

        if seed_models:
            # Add seed models
            for model in seed_models:
                self.population.append(copy.deepcopy(model))

        # Fill remaining slots with new random models
        remaining = self.population_size - len(self.population)
        if remaining > 0:

            new_models = ArchitectureRegistry.create_population(
                remaining,
                use_cuda=self.use_cuda,
            )
            self.population.extend(new_models)

        # Initialize ELO ratings
        self.elo_system.reset_ratings()
        self.elo_ratings = [self.elo_system.initial_rating] * len(self.population)

        print(f"Initialized population of {len(self.population)} models")
        if self.architecture_enabled:
            arch_counts = ArchitectureRegistry.count_architectures(self.population)
            print(f"  Architecture distribution: {arch_counts}")
        print(f"Using {self.num_workers} parallel workers")
        print(f"Games per pairing: {self.games_per_pairing}")
        print(f"CUDA available: {torch.cuda.is_available()}")

    def play_single_hand(
        self,
        models: List[BasicEuchreNN],
        model_indices: List[int],
    ) -> Dict:
        """
        Play a single hand (5 tricks) with specified models.
        Track individual trick winners for per-model rewards.

        Args:
            models: List of 4 models (positions 0-3)
            model_indices: List of 4 model indices in population

        Returns:
            Dictionary with hand results:
            - tricks_won: Dict[model_index -> number of tricks won (0-5)]
            - trump_caller_position: Position who called trump (0-3)
            - trump_caller_index: Model index who called trump
            - calling_team_won: Whether calling team got >=3 tricks
            - team1_tricks: Tricks won by team 1 (positions 0,2)
            - team2_tricks: Tricks won by team 2 (positions 1,3)
        """
        # Create game
        game = EuchreGame(f"training-hand-{random.randint(1000, 9999)}")

        # Add players (all AI for training)
        for i in range(4):
            game.add_player(f"Player{i}", PlayerType.RANDOM_AI)

        # Track results
        trump_caller_position = None
        tricks_won_by_position = {0: 0, 1: 0, 2: 0, 3: 0}

        # Start hand
        game.start_new_hand()

        # Play until hand complete
        while game.state.phase not in [GamePhase.HAND_COMPLETE, GamePhase.GAME_OVER]:
            current_pos = game.state.current_player_position
            current_model = models[current_pos]

            # Handle different game phases
            if game.state.phase == GamePhase.TRUMP_SELECTION_ROUND1:
                # Use neural network for trump selection (Round 1) with masking
                game_state_dict = game.get_state(perspective_position=current_pos)
                turned_up_card = (
                    str(game.state.turned_up_card)
                    if game.state.turned_up_card
                    else None
                )
                trump_state = encode_trump_state(game_state_dict, turned_up_card)

                # Round 1: Only 2 valid options - call turned-up suit or pass
                # Map turned-up suit to index (C=0, D=1, H=2, S=3)
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
                    # Valid options: turned-up suit index and pass (4)
                    valid_trump_indices = [turned_up_suit_idx, 4]
                    decision_idx = current_model.predict_trump_decision_masked(
                        trump_state, valid_trump_indices
                    )
                else:
                    decision_idx = current_model.predict_trump_decision(trump_state)

                if decision_idx == 4:
                    # Model wants to pass
                    try:
                        game.pass_trump()
                    except:
                        # Dealer must call
                        if game.state.turned_up_card:
                            game.call_trump(game.state.turned_up_card.suit)
                            trump_caller_position = current_pos
                else:
                    # Model wants to call
                    try:
                        if game.state.turned_up_card:
                            game.call_trump(game.state.turned_up_card.suit)
                            trump_caller_position = current_pos
                    except:
                        try:
                            game.pass_trump()
                        except:
                            if game.state.turned_up_card:
                                game.call_trump(game.state.turned_up_card.suit)
                                trump_caller_position = current_pos

            elif game.state.phase == GamePhase.TRUMP_SELECTION_ROUND2:
                # Use neural network for trump selection (Round 2) with masking
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

                # Round 2: Can call any suit EXCEPT turned-up, or pass (dealer can't pass)
                suit_map_enum = {
                    Suit.CLUBS: 0,
                    Suit.DIAMONDS: 1,
                    Suit.HEARTS: 2,
                    Suit.SPADES: 3,
                }
                turned_up_suit_idx = suit_map_enum.get(turned_up_suit, 0)

                # Valid options: all suits except turned-up, plus pass (if not dealer)
                valid_trump_indices = [i for i in range(4) if i != turned_up_suit_idx]
                if not is_dealer:
                    valid_trump_indices.append(4)  # Non-dealer can pass

                decision_idx = current_model.predict_trump_decision_masked(
                    trump_state, valid_trump_indices
                )

                if decision_idx == 4:
                    # Try to pass
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
                        trump_caller_position = current_pos
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
                            trump_caller_position = current_pos
                else:
                    # Call the selected suit
                    suit_map = [Suit.CLUBS, Suit.DIAMONDS, Suit.HEARTS, Suit.SPADES]
                    selected_suit = suit_map[decision_idx]

                    if selected_suit != turned_up_suit:
                        game.call_trump(selected_suit)
                        trump_caller_position = current_pos
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
                            trump_caller_position = current_pos
                        else:
                            try:
                                game.pass_trump()
                            except:
                                game.call_trump(random.choice(available_suits))
                                trump_caller_position = current_pos

            elif game.state.phase == GamePhase.DEALER_DISCARD:
                # Use neural network for dealer discard with masking
                dealer = game.state.get_player(game.state.dealer_position)
                game_state_dict = game.get_state(
                    perspective_position=game.state.dealer_position
                )
                hand_with_pickup = [str(card) for card in dealer.hand]
                discard_state = encode_discard_state(game_state_dict, hand_with_pickup)

                # Map dealer's hand to valid indices
                valid_discard_indices = []
                for card in dealer.hand:
                    card_str = str(card)
                    if card_str in self.all_cards:
                        valid_discard_indices.append(self.all_cards.index(card_str))

                if valid_discard_indices:
                    card_idx = current_model.predict_discard_masked(
                        discard_state, valid_discard_indices
                    )
                    predicted_card_str = (
                        self.all_cards[card_idx]
                        if card_idx < len(self.all_cards)
                        else None
                    )

                    card_to_discard = None
                    if predicted_card_str:
                        for card in dealer.hand:
                            if str(card) == predicted_card_str:
                                card_to_discard = card
                                break

                    if card_to_discard is None:
                        card_to_discard = dealer.hand[0]
                else:
                    card_to_discard = dealer.hand[0]

                game.dealer_discard(card_to_discard)

            elif game.state.phase == GamePhase.PLAYING:
                # Use neural network to select card with masking
                game_state_dict = game.get_state(perspective_position=current_pos)
                state_encoding = encode_game_state(game_state_dict)
                valid_cards = game.get_valid_moves(current_pos)

                if valid_cards:
                    # Map valid cards to indices
                    valid_card_indices = []
                    for card in valid_cards:
                        card_str = str(card)
                        if card_str in self.all_cards:
                            valid_card_indices.append(self.all_cards.index(card_str))

                    # Use masked prediction to only consider valid cards
                    if valid_card_indices:
                        card_idx = current_model.predict_card_masked(
                            state_encoding, valid_card_indices
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
                    else:
                        selected_card = valid_cards[0]

                    result = game.play_card(selected_card)

                    # Track trick winner
                    if result.get("trick_complete"):
                        winner_pos = result["trick_winner"]
                        tricks_won_by_position[winner_pos] += 1

        # Calculate final results
        team1_tricks = tricks_won_by_position[0] + tricks_won_by_position[2]
        team2_tricks = tricks_won_by_position[1] + tricks_won_by_position[3]

        # Determine if calling team won (>=3 tricks)
        calling_team = None
        calling_team_won = False
        if trump_caller_position is not None:
            calling_team = 1 if trump_caller_position in [0, 2] else 2
            calling_team_won = (calling_team == 1 and team1_tricks >= 3) or (
                calling_team == 2 and team2_tricks >= 3
            )

        # Map position tricks to model index tricks
        tricks_won_by_index = {
            model_indices[pos]: tricks_won_by_position[pos] for pos in range(4)
        }

        trump_caller_index = None
        if trump_caller_position is not None:
            trump_caller_index = model_indices[trump_caller_position]

        # Calculate Euchre points earned by each team
        # Determine who called trump and calculate points
        points_team1 = 0
        points_team2 = 0

        if calling_team is not None:
            if calling_team_won:
                # Calling team made it (>=3 tricks)
                if calling_team == 1:
                    if team1_tricks == 5:
                        points_team1 = 2  # March
                    else:
                        points_team1 = 1  # Made it
                else:  # calling_team == 2
                    if team2_tricks == 5:
                        points_team2 = 2  # March
                    else:
                        points_team2 = 1  # Made it
            else:
                # Calling team got euchred - opponent gets 2 points
                if calling_team == 1:
                    points_team2 = 2  # Team 2 euchred team 1
                else:
                    points_team1 = 2  # Team 1 euchred team 2

        # Map points to individual players
        # Each player on winning team gets the points, losing team gets negative
        points_by_index = {}
        for pos in range(4):
            idx = model_indices[pos]
            if pos % 2 == 0:  # Team 1
                points_by_index[idx] = points_team1 - points_team2
            else:  # Team 2
                points_by_index[idx] = points_team2 - points_team1

        return {
            "tricks_won": tricks_won_by_index,
            "trump_caller_position": trump_caller_position,
            "trump_caller_index": trump_caller_index,
            "calling_team_won": calling_team_won,
            "team1_tricks": team1_tricks,
            "team2_tricks": team2_tricks,
            "points_earned": points_by_index,
        }

    def play_game_with_teams(
        self,
        team1_models: Tuple[BasicEuchreNN, BasicEuchreNN],
        team2_models: Tuple[BasicEuchreNN, BasicEuchreNN],
    ) -> Dict:
        """
        Play a single game with specified teams using neural networks for all decisions.

        Args:
            team1_models: Tuple of (model_pos0, model_pos2) for team 1
            team2_models: Tuple of (model_pos1, model_pos3) for team 2

        Returns:
            Dictionary with game results:
            - winning_team: 1 or 2
            - team1_score: final score
            - team2_score: final score
            - hands_played: list of hand results with trump calling info
        """
        # Create game
        game = EuchreGame(f"training-{random.randint(1000, 9999)}")

        # Add players (all AI for training)
        for i in range(4):
            game.add_player(f"Player{i}", PlayerType.RANDOM_AI)

        # Map positions to models
        # Positions: 0, 2 = Team 1; 1, 3 = Team 2
        models = [team1_models[0], team2_models[0], team1_models[1], team2_models[1]]

        # Track hand-level statistics for reward shaping
        hands_played = []
        current_hand_info = {
            "caller_position": None,
            "calling_team": None,
            "was_forced_call": False,
            "trump_opportunities": {
                0: 0,
                1: 0,
                2: 0,
                3: 0,
            },  # Per-position opportunities
            "voluntary_calls": {0: 0, 1: 0, 2: 0, 3: 0},  # Per-position voluntary calls
        }

        # Start first hand
        game.start_new_hand()

        # Play until game over
        while game.state.phase != GamePhase.GAME_OVER:
            current_pos = game.state.current_player_position
            current_model = models[current_pos]

            # Handle different game phases
            if game.state.phase == GamePhase.TRUMP_SELECTION_ROUND1:
                # Track trump opportunity for this position
                current_hand_info["trump_opportunities"][current_pos] += 1

                # Use neural network for trump selection
                # ROUND 1: Can only call the turned-up suit or pass
                game_state_dict = game.get_state(perspective_position=current_pos)
                turned_up_card = (
                    str(game.state.turned_up_card)
                    if game.state.turned_up_card
                    else None
                )
                trump_state = encode_trump_state(game_state_dict, turned_up_card)

                decision_idx = current_model.predict_trump_decision(trump_state)

                # FIXED LOGIC: In Round 1, interpret output as binary call/pass
                # decision_idx: 0-3 = CALL (accept turned-up suit), 4 = PASS
                if decision_idx == 4:
                    # Model wants to pass
                    try:
                        game.pass_trump()
                    except:
                        # Dealer must call - use turned up suit
                        if game.state.turned_up_card:
                            game.call_trump(game.state.turned_up_card.suit)
                            # Track: forced call (dealer had to call)
                            current_hand_info["caller_position"] = current_pos
                            current_hand_info["calling_team"] = (
                                1 if current_pos in [0, 2] else 2
                            )
                            current_hand_info["was_forced_call"] = True
                else:
                    # Model wants to call (any output 0-3 means "call")
                    # In Round 1, we can only call the turned-up suit
                    try:
                        if game.state.turned_up_card:
                            game.call_trump(game.state.turned_up_card.suit)
                            # Track: voluntary call in round 1
                            current_hand_info["caller_position"] = current_pos
                            current_hand_info["calling_team"] = (
                                1 if current_pos in [0, 2] else 2
                            )
                            current_hand_info["was_forced_call"] = False
                            current_hand_info["voluntary_calls"][current_pos] += 1
                        else:
                            # Shouldn't happen, but pass if no turned up card
                            game.pass_trump()
                    except:
                        # If call fails, try to pass
                        try:
                            game.pass_trump()
                        except:
                            # Dealer must call
                            if game.state.turned_up_card:
                                game.call_trump(game.state.turned_up_card.suit)
                                current_hand_info["caller_position"] = current_pos
                                current_hand_info["calling_team"] = (
                                    1 if current_pos in [0, 2] else 2
                                )
                                current_hand_info["was_forced_call"] = True

            elif game.state.phase == GamePhase.TRUMP_SELECTION_ROUND2:
                # Track trump opportunity for this position (round 2)
                current_hand_info["trump_opportunities"][current_pos] += 1

                # Use neural network for trump selection (round 2)
                game_state_dict = game.get_state(perspective_position=current_pos)
                turned_up_card = (
                    str(game.state.turned_up_card)
                    if game.state.turned_up_card
                    else None
                )
                trump_state = encode_trump_state(game_state_dict, turned_up_card)

                decision_idx = current_model.predict_trump_decision(trump_state)

                turned_up_suit = (
                    game.state.turned_up_card.suit
                    if game.state.turned_up_card
                    else Suit.CLUBS
                )
                is_dealer = (
                    game.state.current_player_position == game.state.dealer_position
                )

                if decision_idx == 4:
                    # Try to pass
                    if is_dealer:
                        # Dealer must call - pick a suit that's not turned up
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
                        # Track: forced dealer call in round 2 (VERY passive)
                        current_hand_info["caller_position"] = current_pos
                        current_hand_info["calling_team"] = (
                            1 if current_pos in [0, 2] else 2
                        )
                        current_hand_info["was_forced_call"] = True
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
                            current_hand_info["was_forced_call"] = True
                else:
                    # Call the selected suit (must not be turned up suit)
                    suit_map = [Suit.CLUBS, Suit.DIAMONDS, Suit.HEARTS, Suit.SPADES]
                    selected_suit = suit_map[decision_idx]

                    if selected_suit != turned_up_suit:
                        game.call_trump(selected_suit)
                        # Track: voluntary call in round 2
                        current_hand_info["caller_position"] = current_pos
                        current_hand_info["calling_team"] = (
                            1 if current_pos in [0, 2] else 2
                        )
                        current_hand_info["was_forced_call"] = False
                        current_hand_info["voluntary_calls"][current_pos] += 1
                    else:
                        # Invalid choice, pick different suit
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
                        else:
                            try:
                                game.pass_trump()
                            except:
                                game.call_trump(random.choice(available_suits))
                                current_hand_info["caller_position"] = current_pos
                                current_hand_info["calling_team"] = (
                                    1 if current_pos in [0, 2] else 2
                                )
                                current_hand_info["was_forced_call"] = True

            elif game.state.phase == GamePhase.DEALER_DISCARD:
                # Use neural network for dealer discard
                dealer = game.state.get_player(game.state.dealer_position)
                game_state_dict = game.get_state(
                    perspective_position=game.state.dealer_position
                )

                # Hand includes the picked up card
                hand_with_pickup = [str(card) for card in dealer.hand]
                discard_state = encode_discard_state(game_state_dict, hand_with_pickup)

                card_idx = current_model.predict_discard(discard_state)

                # Map to actual card
                if card_idx < len(self.all_cards):
                    predicted_card_str = self.all_cards[card_idx]

                    # Find card in hand
                    card_to_discard = None
                    for card in dealer.hand:
                        if str(card) == predicted_card_str:
                            card_to_discard = card
                            break

                    # If not found, discard first card
                    if card_to_discard is None:
                        card_to_discard = dealer.hand[0]
                else:
                    card_to_discard = dealer.hand[0]

                game.dealer_discard(card_to_discard)

            elif game.state.phase == GamePhase.PLAYING:
                # Use neural network to select card with masking
                game_state_dict = game.get_state(perspective_position=current_pos)
                state_encoding = encode_game_state(game_state_dict)

                # Get valid moves
                valid_cards = game.get_valid_moves(current_pos)

                if valid_cards:
                    # Map valid cards to indices
                    valid_card_indices = []
                    for card in valid_cards:
                        card_str = str(card)
                        if card_str in self.all_cards:
                            valid_card_indices.append(self.all_cards.index(card_str))

                    # Use masked prediction to only consider valid cards
                    if valid_card_indices:
                        card_idx = current_model.predict_card_masked(
                            state_encoding, valid_card_indices
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
                    else:
                        selected_card = valid_cards[0]

                    # Play the card
                    result = game.play_card(selected_card)

                    # Check if hand complete
                    if result.get("hand_complete"):
                        # Start new hand if game not over
                        if game.state.phase != GamePhase.GAME_OVER:
                            game.start_new_hand()

            elif game.state.phase == GamePhase.HAND_COMPLETE:
                # Start new hand
                game.start_new_hand()

        # Return results as dictionary with extended info for reward shaping
        team1_score = game.state.team1_score
        team2_score = game.state.team2_score
        winning_team = 1 if team1_score >= 10 else 2

        # Store final hand info
        hands_played.append(copy.deepcopy(current_hand_info))

        # Determine if euchre occurred (calling team lost)
        calling_team = current_hand_info.get("calling_team")
        was_euchred = calling_team is not None and calling_team != winning_team

        return {
            "winning_team": winning_team,
            "team1_score": team1_score,
            "team2_score": team2_score,
            "calling_team": calling_team,
            "was_forced_call": current_hand_info.get("was_forced_call", False),
            "was_euchred": was_euchred,
            "trump_opportunities": current_hand_info.get("trump_opportunities", {}),
            "voluntary_calls": current_hand_info.get("voluntary_calls", {}),
        }

    def run_round_robin_tournament_elo(
        self, models: List[BasicEuchreNN], model_indices: List[int] | None = None
    ) -> Dict[int, float]:
        """
        Run round-robin tournament with 4 models using ELO ratings.

        Team pairings:
        - Round 1: (A+B) vs (C+D)
        - Round 2: (A+C) vs (B+D)
        - Round 3: (A+D) vs (B+C)

        Args:
            models: List of exactly 4 models
            model_indices: Optional list of indices in population (for ELO tracking)

        Returns:
            Dictionary of model_index -> ELO rating change
        """
        if len(models) != 4:
            raise ValueError("Round-robin tournament requires exactly 4 models")

        if model_indices is None:
            model_indices = list(range(4))

        # Track ELO changes
        elo_changes = {idx: 0.0 for idx in model_indices}

        # Define team pairings for round-robin
        pairings = [
            # Round 1: (A+B) vs (C+D)
            ((models[0], models[1]), (models[2], models[3]), ([0, 1], [2, 3])),
            # Round 2: (A+C) vs (B+D)
            ((models[0], models[2]), (models[1], models[3]), ([0, 2], [1, 3])),
            # Round 3: (A+D) vs (B+C)
            ((models[0], models[3]), (models[1], models[2]), ([0, 3], [1, 2])),
        ]

        # Play games for each pairing
        for pairing_idx, (team1, team2, (team1_local, team2_local)) in enumerate(
            pairings
        ):
            for game_num in range(self.games_per_pairing):
                # Play game and get extended results
                game_result = self.play_game_with_teams(team1, team2)

                winning_team = game_result["winning_team"]
                team1_score = game_result["team1_score"]
                team2_score = game_result["team2_score"]
                calling_team = game_result["calling_team"]
                was_forced_call = game_result["was_forced_call"]
                was_euchred = game_result["was_euchred"]

                # Update ELO ratings for this game
                team1_indices = [model_indices[i] for i in team1_local]
                team2_indices = [model_indices[i] for i in team2_local]

                # Calculate Margin of Victory (MOV) multiplier
                score_diff = abs(team1_score - team2_score)
                mov_multiplier = 1.0 + (score_diff / 20.0)

                # Apply REWARD SHAPING based on trump calling behavior
                reward_multiplier = 1.0
                if self.reward_shaping_enabled and calling_team is not None:
                    if winning_team == calling_team:
                        # Winning team called trump - reward aggressive play!
                        if was_forced_call:
                            # Forced call (dealer had no choice) - small penalty
                            reward_multiplier += self.forced_call_penalty
                        else:
                            # Voluntary call that won - big bonus!
                            reward_multiplier += self.aggressive_win_bonus
                    else:
                        # Defending team won (calling team lost = euchre)
                        # Reward defending team for good defense
                        reward_multiplier += self.euchre_bonus

                # Temporarily adjust K-factor for this update
                original_k = self.elo_system.k_factor
                self.elo_system.k_factor *= mov_multiplier * reward_multiplier

                updated_ratings = self.elo_system.update_team_ratings(
                    team1_indices,
                    team2_indices,
                    team1_won=(winning_team == 1),
                    team1_score=team1_score,
                    team2_score=team2_score,
                )

                # Restore original K-factor
                self.elo_system.k_factor = original_k

                # Track changes
                for idx in team1_indices + team2_indices:
                    if idx in model_indices:
                        local_idx = model_indices.index(idx)
                        elo_changes[idx] = (
                            self.elo_system.get_rating(idx)
                            - self.elo_system.initial_rating
                        )

        return elo_changes

    def run_round_robin_hand_based(
        self, models: List[BasicEuchreNN], model_indices: List[int] | None = None
    ) -> Dict[int, float]:
        """
        Run round-robin hands (not full games) with 4 models using accumulated points.

        ANTI-DEGENERATE STRATEGY:
        - Accumulates points over many hands before updating ELO
        - Penalizes models that call trump too frequently (>80%)
        - Penalizes models with high euchre rates (>40%)
        - Rewards selective, strategic trump calling

        Args:
            models: List of exactly 4 models
            model_indices: Optional list of indices in population (for ELO tracking)

        Returns:
            Dictionary of model_index -> ELO rating change
        """
        if len(models) != 4:
            raise ValueError("Round-robin tournament requires exactly 4 models")

        if model_indices is None:
            model_indices = list(range(4))

        # Track accumulated statistics
        accumulated_points = {idx: 0 for idx in model_indices}
        call_counts = {idx: 0 for idx in model_indices}  # Track calling behavior
        euchre_counts = {idx: 0 for idx in model_indices}  # Track euchres
        trump_opportunities = {
            idx: 0 for idx in model_indices
        }  # Track opportunities to call

        # Play many hands and accumulate statistics
        for hand_num in range(self.games_per_pairing):
            # Play a single hand with all 4 models
            hand_result = self.play_single_hand(models, model_indices)

            # Accumulate points
            for idx, points in hand_result["points_earned"].items():
                accumulated_points[idx] += points

            # Track who called trump and if they got euchred
            if hand_result.get("trump_caller_index") is not None:
                caller_idx = hand_result["trump_caller_index"]
                call_counts[caller_idx] += 1

                # Track if this call resulted in a euchre
                if not hand_result.get("calling_team_won"):
                    euchre_counts[caller_idx] += 1

            # Track trump opportunities (each player gets 1-2 chances per hand)
            # Approximate: each hand has ~1.5 opportunities per player on average
            for idx in model_indices:
                trump_opportunities[idx] += 1

        # ANTI-DEGENERATE PENALTIES
        for idx in model_indices:
            # Calculate call rate (how often they call when they have opportunity)
            call_rate = call_counts[idx] / max(1, trump_opportunities[idx])

            # PENALTY 1: Over-calling (calling >80% of the time)
            if call_rate > 0.8:
                over_call_penalty = int((call_rate - 0.8) * 100)  # Scale penalty
                accumulated_points[idx] -= over_call_penalty
                # Debug logging (optional)
                # print(f"    Model {idx}: Over-calling penalty -{over_call_penalty} (rate: {call_rate:.2%})")

            # PENALTY 2: High euchre rate (getting euchred >40% when calling)
            if call_counts[idx] > 0:
                euchre_rate = euchre_counts[idx] / call_counts[idx]
                if euchre_rate > 0.4:
                    euchre_penalty = int((euchre_rate - 0.4) * 150)  # Stronger penalty
                    accumulated_points[idx] -= euchre_penalty
                    # Debug logging (optional)
                    # print(f"    Model {idx}: High euchre penalty -{euchre_penalty} (rate: {euchre_rate:.2%})")

            # PENALTY 3: Under-calling (never calling, always passing)
            if call_rate < 0.05:  # Calling <5% of the time
                under_call_penalty = int((0.05 - call_rate) * 80)
                accumulated_points[idx] -= under_call_penalty
                # Debug logging (optional)
                # print(f"    Model {idx}: Under-calling penalty -{under_call_penalty} (rate: {call_rate:.2%})")

        # Update ELO once based on final accumulated points
        updated_ratings = self.elo_system.update_individual_ratings(
            model_indices,
            accumulated_points,
        )

        # Track changes
        elo_changes = {idx: 0.0 for idx in model_indices}
        for idx in model_indices:
            elo_changes[idx] = (
                self.elo_system.get_rating(idx) - self.elo_system.initial_rating
            )

        return elo_changes

    def run_sampled_round_robin(self) -> List[float]:
        """
        NEW: Sampled round-robin tournament where each model plays against
        random opponents with shuffled team compositions.

        Uses chess-style ELO updates (head-to-head).
        Parallelized for speed using ThreadPoolExecutor.

        Returns:
            List of ELO ratings for each model
        """
        # Reset ELO system for this generation
        self.elo_system.reset_ratings()

        print(f"Running sampled round-robin tournament (parallel)...")
        print(f"  Each model plays {self.hands_per_model} hands")
        print(f"  Using chess-style ELO updates (head-to-head)")
        print(f"  Workers: {self.num_workers}")

        # Pre-generate all matchups
        all_matchups = []
        for model_idx in range(len(self.population)):
            for hand_num in range(self.hands_per_model):
                # Randomly select 3 other models to play with
                other_indices = [
                    i for i in range(len(self.population)) if i != model_idx
                ]
                selected_indices = random.sample(
                    other_indices, min(3, len(other_indices))
                )

                # If not enough models, fill with duplicates
                while len(selected_indices) < 3:
                    selected_indices.append(random.choice(other_indices))

                # Create team of 4: current model + 3 random others
                all_indices = [model_idx] + selected_indices
                random.shuffle(all_indices)  # Shuffle positions

                all_matchups.append(all_indices)

        total_hands = len(all_matchups)
        print(f"  Total hands to play: {total_hands}")

        # Use parallel processing if enabled
        if self.parallel_mode == "sequential" or self.num_workers <= 1:
            # Sequential processing
            for matchup_indices in all_matchups:
                self._play_and_update_hand(matchup_indices)
        else:
            # Parallel processing with ThreadPoolExecutor
            self._run_matchups_parallel(all_matchups)

        # Get final ELO ratings
        elo_ratings = [
            self.elo_system.get_rating(i) for i in range(len(self.population))
        ]

        return elo_ratings

    def _play_and_update_hand(self, all_indices: List[int]):
        """
        Play a single hand and update ELO ratings.

        Args:
            all_indices: List of 4 model indices for the hand
        """
        # Get models
        models = [self.population[idx] for idx in all_indices]

        # Play a single hand
        hand_result = self.play_single_hand(models, all_indices)

        # Determine winning and losing teams
        team1_won = hand_result["team1_tricks"] >= 3
        team1_indices = [all_indices[0], all_indices[2]]
        team2_indices = [all_indices[1], all_indices[3]]

        # Update ELO using head-to-head method (chess-style)
        if team1_won:
            self.elo_system.update_head_to_head(team1_indices, team2_indices)
        else:
            self.elo_system.update_head_to_head(team2_indices, team1_indices)

    def _run_matchups_parallel(self, all_matchups: List[List[int]]):
        """
        Run matchups in parallel using ThreadPoolExecutor.

        Args:
            all_matchups: List of matchups (each is a list of 4 model indices)
        """
        import threading

        # Create a lock for thread-safe ELO updates
        elo_lock = threading.Lock()
        completed = [0]  # Use list to allow modification in nested function

        def play_matchup_batch(batch):
            """Play a batch of matchups and collect results"""
            results = []
            for matchup_indices in batch:
                # Get models
                models = [self.population[idx] for idx in matchup_indices]

                # Play a single hand
                hand_result = self.play_single_hand(models, matchup_indices)

                # Determine winning and losing teams
                team1_won = hand_result["team1_tricks"] >= 3
                team1_indices = [matchup_indices[0], matchup_indices[2]]
                team2_indices = [matchup_indices[1], matchup_indices[3]]

                results.append((team1_won, team1_indices, team2_indices))

            # Update ELO ratings (thread-safe)
            with elo_lock:
                for team1_won, team1_indices, team2_indices in results:
                    if team1_won:
                        self.elo_system.update_head_to_head(
                            team1_indices, team2_indices
                        )
                    else:
                        self.elo_system.update_head_to_head(
                            team2_indices, team1_indices
                        )

                completed[0] += len(batch)
                if completed[0] % 100 == 0 or completed[0] == len(all_matchups):
                    print(
                        f"  Completed {completed[0]}/{len(all_matchups)} hands ({100*completed[0]//len(all_matchups)}%)"
                    )

        # Divide matchups into batches for workers
        batch_size = max(1, len(all_matchups) // (self.num_workers * 4))
        batches = [
            all_matchups[i : i + batch_size]
            for i in range(0, len(all_matchups), batch_size)
        ]

        print(f"  Running {len(batches)} batches with {self.num_workers} workers...")

        # Run batches in parallel
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [executor.submit(play_matchup_batch, batch) for batch in batches]

            # Wait for all to complete
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"  Error in matchup batch: {e}")
                    import traceback

                    traceback.print_exc()

    def evaluate_population_parallel(self) -> List[float]:
        """
        Evaluate entire population using parallel tournament groups.

        Uses ThreadPoolExecutor for parallel execution (better for I/O bound and
        avoids multiprocessing overhead with PyTorch models).

        Returns:
            List of ELO ratings for each model
        """
        # Use sampled round-robin if using champion_variations strategy
        if self.evolution_strategy == "champion_variations":
            return self.run_sampled_round_robin()

        # Reset ELO system for this generation
        self.elo_system.reset_ratings()

        # Divide population into groups of 4
        num_groups = len(self.population) // 4
        groups = []

        for group_idx in range(num_groups):
            start_idx = group_idx * 4
            group_models = self.population[start_idx : start_idx + 4]
            group_indices = list(range(start_idx, start_idx + 4))
            groups.append((group_models, group_indices))

        # Handle remaining models (if population not divisible by 4)
        remaining = len(self.population) % 4
        if remaining > 0:
            remaining_models = self.population[-remaining:]
            # Get fill models and their indices
            fill_count = 4 - remaining
            fill_indices = random.sample(
                range(len(self.population) - remaining), fill_count
            )
            fill_models = [self.population[i] for i in fill_indices]
            group_models = remaining_models + fill_models
            # Include indices for ALL 4 models (remaining + fill)
            remaining_indices = list(
                range(len(self.population) - remaining, len(self.population))
            )
            group_indices = remaining_indices + fill_indices
            groups.append((group_models, group_indices))

        print(
            f"Running {len(groups)} tournament groups with {self.num_workers} workers..."
        )
        print(f"Parallel mode: {self.parallel_mode}, CUDA enabled: {self.use_cuda}")

        if self.parallel_mode == "sequential" or self.num_workers <= 1:
            # Sequential execution - use hand-based evaluation
            for group_idx, (group_models, group_indices) in enumerate(groups):
                print(
                    f"  Group {group_idx + 1}/{len(groups)}: {self.games_per_pairing} hands per group (individual trick rewards)"
                )
                self.run_round_robin_hand_based(group_models, group_indices)
        elif self.parallel_mode == "thread":
            # Thread-based parallelism (recommended for PyTorch)
            # This avoids the overhead of serializing models for multiprocessing
            self._run_groups_threaded(groups)
        else:
            # Process-based parallelism (for CPU-bound without GPU)
            self._run_groups_multiprocess(groups)

        # Get final ELO ratings
        elo_ratings = [
            self.elo_system.get_rating(i) for i in range(len(self.population))
        ]

        return elo_ratings

    def _run_groups_threaded(self, groups: List[Tuple[List[BasicEuchreNN], List[int]]]):
        """Run tournament groups using thread pool for parallelism."""
        import threading

        # Create a lock for thread-safe ELO updates
        elo_lock = threading.Lock()
        completed = [0]  # Use list to allow modification in nested function

        def run_group(group_data):
            group_models, group_indices = group_data
            # Create a temporary GA instance for this group
            ga_temp = GeneticAlgorithm(
                games_per_pairing=self.games_per_pairing,
                use_cuda=self.use_cuda,
                parallel_mode="sequential",
            )
            ga_temp.all_cards = self.all_cards

            # Initialize ELO system for this group
            for idx in group_indices:
                ga_temp.elo_system.ratings[idx] = ga_temp.elo_system.initial_rating

            # Run round-robin hand-based evaluation (INDIVIDUAL TRICK REWARDS)
            ga_temp.run_round_robin_hand_based(group_models, group_indices)

            # Collect results
            results = {idx: ga_temp.elo_system.get_rating(idx) for idx in group_indices}

            # Update main ELO system (thread-safe)
            with elo_lock:
                for idx, rating in results.items():
                    # Merge ratings - take the delta from initial and apply to main system
                    delta = rating - ga_temp.elo_system.initial_rating
                    current = self.elo_system.get_rating(idx)
                    self.elo_system.ratings[idx] = current + delta

                completed[0] += 1
                print(f"  Completed group {completed[0]}/{len(groups)}")

            return results

        # Run groups in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [executor.submit(run_group, group) for group in groups]

            # Wait for all to complete
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"  Error in tournament group: {e}")

    def _run_groups_multiprocess(
        self, groups: List[Tuple[List[BasicEuchreNN], List[int]]]
    ):
        """Run tournament groups using process pool for parallelism."""
        # Prepare arguments for worker processes
        # We need to serialize model state dicts since PyTorch models can't be pickled directly
        worker_args = []
        for group_models, group_indices in groups:
            state_dicts = [model.state_dict() for model in group_models]
            worker_args.append(
                (
                    state_dicts,
                    self.games_per_pairing,
                    self.all_cards,
                    group_indices,
                    self.use_cuda,
                )
            )

        # Run in parallel using ProcessPoolExecutor
        completed = 0
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [
                executor.submit(run_tournament_group_worker, args)
                for args in worker_args
            ]

            for future in as_completed(futures):
                try:
                    results = future.result()
                    # Update main ELO system with results
                    for idx, rating in results.items():
                        delta = rating - self.elo_system.initial_rating
                        current = self.elo_system.get_rating(idx)
                        self.elo_system.ratings[idx] = current + delta

                    completed += 1
                    print(f"  Completed group {completed}/{len(groups)}")
                except Exception as e:
                    print(f"  Error in tournament group: {e}")
                    import traceback

                    traceback.print_exc()

    def selection(self) -> List[BasicEuchreNN]:
        """
        Select parents for next generation using adaptive tournament selection.
        Uses larger tournament size for stronger selection pressure.
        """
        parents = []

        # Adaptive tournament size based on diversity
        # Larger tournament = more selection pressure = faster convergence to best
        base_tournament_size = 5  # Increased from 3

        # Increase tournament size when stagnating to focus on best models
        if self.generations_without_improvement > 5:
            tournament_size = min(7, len(self.population) // 2)
        else:
            tournament_size = base_tournament_size

        for _ in range(self.population_size - self.elite_size):
            # Tournament selection with adaptive size
            tournament = random.sample(
                list(zip(self.population, self.elo_ratings)),
                min(tournament_size, len(self.population)),
            )
            winner = max(tournament, key=lambda x: x[1])[0]
            parents.append(winner)

        return parents

    def crossover(
        self, parent1: BasicEuchreNN, parent2: BasicEuchreNN
    ) -> BasicEuchreNN:
        """
        Improved crossover using layer-wise blending to preserve good patterns.
        Uses weighted average based on parent fitness rather than random mixing.
        Handles multi-architecture populations by only crossing over same types.
        """
        if random.random() > self.crossover_rate:
            return copy.deepcopy(parent1)

        # Check if architectures match
        arch1 = ArchitectureRegistry.get_architecture_type(parent1)
        arch2 = ArchitectureRegistry.get_architecture_type(parent2)

        if arch1 != arch2:
            # Cannot crossover different architectures, return a copy of parent1
            return copy.deepcopy(parent1)

        # Create child of the same architecture
        child = copy.deepcopy(parent1)

        # Layer-wise crossover with adaptive blending
        for child_param, p1_param, p2_param in zip(
            child.parameters(), parent1.parameters(), parent2.parameters()
        ):
            # Use blend crossover (weighted average) instead of uniform
            # This preserves more structure from both parents
            alpha = random.uniform(0.3, 0.7)  # Blend factor
            child_param.data = alpha * p1_param.data + (1 - alpha) * p2_param.data

        return child

    def mutate(self, model: BasicEuchreNN):
        """
        Improved mutation with adaptive strength and layer-wise probability.
        Uses smaller mutations to preserve good solutions while still exploring.
        """
        for param in model.parameters():
            if random.random() < self.mutation_rate:
                # Use mutation_strength parameter for champion_variations
                mutation_str = (
                    self.mutation_strength
                    if self.evolution_strategy == "champion_variations"
                    else 0.05 * self.temperature
                )

                # Add Gaussian noise to weights
                noise = torch.randn_like(param) * mutation_str
                param.data += noise

                # Clip extreme values to prevent instability
                param.data = torch.clamp(param.data, -10.0, 10.0)

    def create_variations(
        self, champion: BasicEuchreNN, count: int
    ) -> List[BasicEuchreNN]:
        """
        Create variations of the champion model through mutation.

        Used in champion_variations evolution strategy.

        Args:
            champion: The champion model to create variations from
            count: Number of variations to create

        Returns:
            List of mutated variations
        """
        variations = []
        for i in range(count):
            # Create a copy of the champion
            variation = copy.deepcopy(champion)

            # Mutate it
            self.mutate(variation)

            variations.append(variation)

        return variations

    def apply_diversity_boost(self, level: int):
        """
        Apply diversity-boosting mechanisms based on stagnation level.

        Args:
            level: Stagnation level (1, 2, or 3)
        """
        if level == 1:
            # Level 1: Moderate intervention (5 generations stagnant)
            print(f"  🔄 DIVERSITY BOOST LEVEL 1:")
            print(f"     - Increasing mutation rate to {self.mutation_rate:.3f}")
            print(f"     - Injecting 10% random immigrants")

            # Add random immigrants (10% of population)
            num_immigrants = max(1, int(self.population_size * 0.1))
            for i in range(num_immigrants):

                # Create transformer model (best performing architecture)
                new_model = ArchitectureRegistry.create_model(use_cuda=self.use_cuda)

                # Replace a random non-elite member
                replace_idx = random.randint(self.elite_size, len(self.population) - 1)
                self.population[replace_idx] = new_model

        elif level == 2:
            # Level 2: Strong intervention (10 generations stagnant)
            print(f"  🔥 DIVERSITY BOOST LEVEL 2:")
            print(f"     - Increasing mutation rate to {self.mutation_rate:.3f}")
            print(f"     - Replacing 25% of population with new transformers")
            print(f"     - Temperature: {self.temperature:.2f}")

            # Replace 25% of population with new transformer models
            if self.architecture_enabled:
                num_to_switch = max(1, int(self.population_size * 0.25))
                for i in range(num_to_switch):
                    # Pick a random non-elite model
                    idx = random.randint(self.elite_size, len(self.population) - 1)
                    # Create new transformer model (best performing architecture)
                    new_model = ArchitectureRegistry.create_model(
                        use_cuda=self.use_cuda
                    )
                    self.population[idx] = new_model

        elif level >= 3:
            # Level 3: Nuclear option (15+ generations stagnant)
            print(f"  💥 DIVERSITY BOOST LEVEL 3 (NUCLEAR):")
            print(f"     - Keeping only top 10% elite")
            print(f"     - Generating 50% new models")
            print(f"     - Resetting temperature to {self.initial_temperature}")

            # Keep only top 10%
            sorted_pop = sorted(
                zip(self.population, self.elo_ratings),
                key=lambda x: x[1],
                reverse=True,
            )
            elite_count = max(2, int(self.population_size * 0.1))
            elites = [copy.deepcopy(model) for model, _ in sorted_pop[:elite_count]]

            # Generate 50% completely new transformer models
            new_count = int(self.population_size * 0.5)
            new_models = []

            new_models = ArchitectureRegistry.create_population(
                new_count,
                use_cuda=self.use_cuda,
            )

            # Fill remaining with mutated elites
            remaining = self.population_size - len(elites) - len(new_models)
            mutated_elites = []
            for _ in range(remaining):
                parent = copy.deepcopy(random.choice(elites))
                self.mutate(parent)
                mutated_elites.append(parent)

            # Rebuild population
            self.population = elites + new_models + mutated_elites
            random.shuffle(self.population)

            # Reset temperature
            self.temperature = self.initial_temperature

    def update_temperature(self, improved: bool):
        """
        Update simulated annealing temperature based on progress.

        Args:
            improved: Whether the best model improved this generation
        """
        if improved:
            # Cool down when making progress
            self.temperature *= self.cooling_rate
            self.temperature = max(0.1, self.temperature)  # Min temperature
        else:
            # Heat up when stagnating
            self.temperature *= self.heating_rate
            self.temperature = min(
                self.max_temperature, self.temperature
            )  # Max temperature

    def update_architecture_stats(self):
        """Track performance statistics by architecture type."""
        if not self.architecture_enabled:
            return

        # Count architectures and their average ELO
        arch_counts = ArchitectureRegistry.count_architectures(self.population)
        arch_elos = {
            arch: [] for arch in ArchitectureRegistry.get_available_architectures()
        }

        for model, elo in zip(self.population, self.elo_ratings):
            arch_type = ArchitectureRegistry.get_architecture_type(model)
            if arch_type in arch_elos:
                arch_elos[arch_type].append(elo)

        # Calculate average ELO per architecture
        self.architecture_stats = {}
        for arch_type in arch_elos:
            if arch_elos[arch_type]:
                avg_elo = sum(arch_elos[arch_type]) / len(arch_elos[arch_type])
                self.architecture_stats[arch_type] = {
                    "count": arch_counts.get(arch_type, 0),
                    "avg_elo": avg_elo,
                    "max_elo": max(arch_elos[arch_type]),
                }

    def print_architecture_stats(self):
        """Print architecture statistics."""
        if not self.architecture_enabled or not self.architecture_stats:
            return

        print(f"\n  Architecture Distribution:")
        for arch_type, stats in self.architecture_stats.items():
            arch_info = ArchitectureRegistry.get_architecture_info(arch_type)
            print(
                f"    {arch_info['name']:12} - Count: {stats['count']:2}, "
                f"Avg ELO: {stats['avg_elo']:6.0f}, Max ELO: {stats['max_elo']:6.0f}"
            )

    def evolve(
        self,
        generations=10,
        callback=None,
        seed_models: List[BasicEuchreNN] | None = None,
    ):
        """
        Run the genetic algorithm for specified generations.

        Args:
            generations: Number of generations to evolve
            callback: Optional callback function called each generation with (gen, best_elo, avg_elo)
            seed_models: Optional list of models to seed the initial population

        Returns:
            Best model from final generation
        """
        self.initialize_population(seed_models)

        for generation in range(generations):
            print(f"\n{'='*60}")
            print(f"Generation {generation + 1}/{generations}")
            print(f"{'='*60}")

            # Evaluate population using ELO ratings
            self.elo_ratings = self.evaluate_population_parallel()

            # Sort by ELO rating
            sorted_pop = sorted(
                zip(self.population, self.elo_ratings),
                key=lambda x: x[1],
                reverse=True,
            )

            current_best_model, current_best_elo = sorted_pop[0]
            avg_elo = sum(self.elo_ratings) / len(self.elo_ratings)

            # Update global champion if current best is better
            prev_best_elo = self.global_best_elo
            improved = current_best_elo > self.global_best_elo

            if improved:
                self.global_best_model = copy.deepcopy(current_best_model)
                self.global_best_elo = current_best_elo
                self.generations_without_improvement = 0
                print(f"\n🏆 NEW GLOBAL CHAMPION! ELO: {current_best_elo:.0f}")
            else:
                self.generations_without_improvement += 1

            # Update simulated annealing temperature
            self.update_temperature(improved)

            # Apply diversity boosts based on stagnation level
            if self.generations_without_improvement == 5:
                self.mutation_rate = min(self.base_mutation_rate * 1.5, 0.3)
                self.apply_diversity_boost(1)
            elif self.generations_without_improvement == 10:
                self.mutation_rate = min(self.base_mutation_rate * 2.0, 0.4)
                self.apply_diversity_boost(2)
            elif self.generations_without_improvement >= 15:
                self.mutation_rate = min(self.base_mutation_rate * 2.5, 0.5)
                self.apply_diversity_boost(3)
                # Reset counter after nuclear option
                self.generations_without_improvement = 0
            elif self.generations_without_improvement < 5:
                # Reset mutation rate when making progress
                self.mutation_rate = self.base_mutation_rate

            # Update architecture statistics
            self.update_architecture_stats()

            print(f"\nGeneration {generation + 1} Summary:")
            print(f"  Current Best ELO:  {current_best_elo:.0f}")
            print(f"  Global Best ELO:   {self.global_best_elo:.0f}")
            print(f"  Average ELO:       {avg_elo:.0f}")
            print(f"  Mutation Rate:     {self.mutation_rate:.3f}")
            print(f"  Temperature:       {self.temperature:.2f}")
            print(f"  Gens w/o Improve:  {self.generations_without_improvement}")

            # Print architecture stats
            self.print_architecture_stats()

            # Call callback if provided (use global best for tracking)
            if callback:
                callback(generation + 1, self.global_best_elo, avg_elo)

            # CHAMPION VARIATIONS STRATEGY: Single champion + variations
            if self.evolution_strategy == "champion_variations":
                # Keep the champion (best model)
                champion = copy.deepcopy(current_best_model)
                print(f"  ✓ Champion preserved (ELO: {current_best_elo:.0f})")

                # Create variations of the champion
                variations = self.create_variations(champion, self.population_size - 1)

                # New population = champion + variations
                self.population = [champion] + variations
                print(f"  Created {len(variations)} variations of champion")

            else:
                # TRADITIONAL GENETIC ALGORITHM STRATEGY
                # Keep elite - ALWAYS include global champion first
                elites = []
                if self.global_best_model is not None:
                    elites.append(copy.deepcopy(self.global_best_model))
                    print(f"  ✓ Global champion preserved in population")

                # Add remaining elites from current generation (avoid duplicates)
                for model, elo in sorted_pop[: self.elite_size]:
                    if len(elites) < self.elite_size:
                        elites.append(copy.deepcopy(model))

                # Select parents
                parents = self.selection()

                # Create offspring
                offspring = []
                for i in range(0, len(parents), 2):
                    if i + 1 < len(parents):
                        child1 = self.crossover(parents[i], parents[i + 1])
                        child2 = self.crossover(parents[i + 1], parents[i])
                        self.mutate(child1)
                        self.mutate(child2)
                        offspring.extend([child1, child2])

                # New population
                self.population = (
                    elites + offspring[: self.population_size - self.elite_size]
                )

        # Return best model
        final_sorted = sorted(
            zip(self.population, self.elo_ratings), key=lambda x: x[1], reverse=True
        )
        return final_sorted[0][0]
