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
        population_size=40,
        mutation_rate=0.15,
        crossover_rate=0.7,
        elite_size=4,
        games_per_pairing=50,
        num_workers=None,
        use_elo=True,
        use_cuda=None,
        parallel_mode="thread",  # "thread", "process", or "sequential"
    ):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.base_mutation_rate = mutation_rate  # Store base for adaptive mutation
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
        self.games_per_pairing = games_per_pairing
        self.num_workers = num_workers or max(1, multiprocessing.cpu_count() - 1)
        self.use_elo = use_elo
        self.parallel_mode = parallel_mode
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
        self.architecture_enabled = True  # Enable multi-architecture support
        self.architecture_stats = {}  # Track performance by architecture

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
            if self.architecture_enabled:
                # Create diverse population with all architectures
                new_models = ArchitectureRegistry.create_population(
                    remaining, use_cuda=self.use_cuda
                )
                self.population.extend(new_models)
            else:
                # Create basic models only
                while len(self.population) < self.population_size:
                    self.population.append(BasicEuchreNN(use_cuda=self.use_cuda))

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

    def play_game_with_teams(
        self,
        team1_models: Tuple[BasicEuchreNN, BasicEuchreNN],
        team2_models: Tuple[BasicEuchreNN, BasicEuchreNN],
    ) -> Tuple[int, int, int]:
        """
        Play a single game with specified teams using neural networks for all decisions.

        Args:
            team1_models: Tuple of (model_pos0, model_pos2) for team 1
            team2_models: Tuple of (model_pos1, model_pos3) for team 2

        Returns:
            Tuple of (winning_team, team1_score, team2_score)
        """
        # Create game
        game = EuchreGame(f"training-{random.randint(1000, 9999)}")

        # Add players (all AI for training)
        for i in range(4):
            game.add_player(f"Player{i}", PlayerType.RANDOM_AI)

        # Map positions to models
        models = [team1_models[0], team2_models[0], team1_models[1], team2_models[1]]

        # Start first hand
        game.start_new_hand()

        # Play until game over
        while game.state.phase != GamePhase.GAME_OVER:
            current_pos = game.state.current_player_position
            current_model = models[current_pos]

            # Handle different game phases
            if game.state.phase == GamePhase.TRUMP_SELECTION_ROUND1:
                # Use neural network for trump selection
                game_state_dict = game.get_state(perspective_position=current_pos)
                turned_up_card = (
                    str(game.state.turned_up_card)
                    if game.state.turned_up_card
                    else None
                )
                trump_state = encode_trump_state(game_state_dict, turned_up_card)

                decision_idx = current_model.predict_trump_decision(trump_state)

                # decision_idx: 0-3 = call suits (C,D,H,S), 4 = pass
                if decision_idx == 4:
                    # Try to pass
                    try:
                        game.pass_trump()
                    except:
                        # Dealer must call - use turned up suit
                        if game.state.turned_up_card:
                            game.call_trump(game.state.turned_up_card.suit)
                else:
                    # Call the selected suit
                    suit_map = [Suit.CLUBS, Suit.DIAMONDS, Suit.HEARTS, Suit.SPADES]
                    selected_suit = suit_map[decision_idx]

                    # Check if this is the turned up suit (valid in round 1)
                    if (
                        game.state.turned_up_card
                        and selected_suit == game.state.turned_up_card.suit
                    ):
                        game.call_trump(selected_suit)
                    else:
                        # Can't call different suit in round 1, so pass
                        try:
                            game.pass_trump()
                        except:
                            # Dealer must call
                            if game.state.turned_up_card:
                                game.call_trump(game.state.turned_up_card.suit)

            elif game.state.phase == GamePhase.TRUMP_SELECTION_ROUND2:
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

                if decision_idx == 4:
                    # Try to pass
                    if game.state.current_player_position == game.state.dealer_position:
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
                    else:
                        try:
                            game.pass_trump()
                        except:
                            # Shouldn't happen but handle it
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
                else:
                    # Call the selected suit (must not be turned up suit)
                    suit_map = [Suit.CLUBS, Suit.DIAMONDS, Suit.HEARTS, Suit.SPADES]
                    selected_suit = suit_map[decision_idx]

                    if selected_suit != turned_up_suit:
                        game.call_trump(selected_suit)
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
                        if (
                            game.state.current_player_position
                            == game.state.dealer_position
                        ):
                            game.call_trump(random.choice(available_suits))
                        else:
                            try:
                                game.pass_trump()
                            except:
                                game.call_trump(random.choice(available_suits))

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
                # Use neural network to select card
                game_state_dict = game.get_state(perspective_position=current_pos)
                state_encoding = encode_game_state(game_state_dict)

                # Get valid moves
                valid_cards = game.get_valid_moves(current_pos)

                if valid_cards:
                    # Get model prediction
                    card_idx = current_model.predict_card(state_encoding)
                    predicted_card_str = (
                        self.all_cards[card_idx]
                        if card_idx < len(self.all_cards)
                        else None
                    )

                    # Check if predicted card is valid
                    selected_card = None
                    if predicted_card_str:
                        for card in valid_cards:
                            if str(card) == predicted_card_str:
                                selected_card = card
                                break

                    # If prediction not valid, pick first valid card
                    if selected_card is None:
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

        # Return results
        team1_score = game.state.team1_score
        team2_score = game.state.team2_score
        winning_team = 1 if team1_score >= 10 else 2

        return winning_team, team1_score, team2_score

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
                winning_team, team1_score, team2_score = self.play_game_with_teams(
                    team1, team2
                )

                # Update ELO ratings for this game
                team1_indices = [model_indices[i] for i in team1_local]
                team2_indices = [model_indices[i] for i in team2_local]

                updated_ratings = self.elo_system.update_team_ratings(
                    team1_indices,
                    team2_indices,
                    team1_won=(winning_team == 1),
                    team1_score=team1_score,
                    team2_score=team2_score,
                )

                # Track changes
                for idx in team1_indices + team2_indices:
                    if idx in model_indices:
                        local_idx = model_indices.index(idx)
                        elo_changes[idx] = (
                            self.elo_system.get_rating(idx)
                            - self.elo_system.initial_rating
                        )

        return elo_changes

    def evaluate_population_parallel(self) -> List[float]:
        """
        Evaluate entire population using parallel tournament groups.

        Uses ThreadPoolExecutor for parallel execution (better for I/O bound and
        avoids multiprocessing overhead with PyTorch models).

        Returns:
            List of ELO ratings for each model
        """
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
            # Sequential execution (original behavior)
            for group_idx, (group_models, group_indices) in enumerate(groups):
                print(
                    f"  Group {group_idx + 1}/{len(groups)}: {self.games_per_pairing * 3} games per model"
                )
                self.run_round_robin_tournament_elo(group_models, group_indices)
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

            # Run round-robin tournament
            ga_temp.run_round_robin_tournament_elo(group_models, group_indices)

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
        """Select parents for next generation using tournament selection based on ELO"""
        parents = []

        for _ in range(self.population_size - self.elite_size):
            # Tournament selection
            tournament_size = 3
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
        """Crossover two parent networks to create offspring"""
        if random.random() > self.crossover_rate:
            return copy.deepcopy(parent1)

        child = BasicEuchreNN()

        # Crossover weights
        for child_param, p1_param, p2_param in zip(
            child.parameters(), parent1.parameters(), parent2.parameters()
        ):
            # Uniform crossover
            mask = torch.rand_like(child_param) > 0.5
            child_param.data = torch.where(mask, p1_param.data, p2_param.data)

        return child

    def mutate(self, model: BasicEuchreNN):
        """Mutate model weights with adaptive mutation rate"""
        for param in model.parameters():
            if random.random() < self.mutation_rate:
                # Add Gaussian noise to weights
                noise = torch.randn_like(param) * 0.1
                param.data += noise

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
                if self.architecture_enabled:
                    # Create random model with random architecture
                    new_model = ArchitectureRegistry.create_random_model(self.use_cuda)
                else:
                    new_model = BasicEuchreNN(use_cuda=self.use_cuda)

                # Replace a random non-elite member
                replace_idx = random.randint(self.elite_size, len(self.population) - 1)
                self.population[replace_idx] = new_model

        elif level == 2:
            # Level 2: Strong intervention (10 generations stagnant)
            print(f"  🔥 DIVERSITY BOOST LEVEL 2:")
            print(f"     - Increasing mutation rate to {self.mutation_rate:.3f}")
            print(f"     - Architecture switching for 25% of population")
            print(f"     - Temperature: {self.temperature:.2f}")

            # Switch architectures for 25% of population
            if self.architecture_enabled:
                num_to_switch = max(1, int(self.population_size * 0.25))
                for i in range(num_to_switch):
                    # Pick a random non-elite model
                    idx = random.randint(self.elite_size, len(self.population) - 1)
                    # Create new model with different architecture
                    new_model = ArchitectureRegistry.create_random_model(self.use_cuda)
                    self.population[idx] = new_model

        elif level >= 3:
            # Level 3: Nuclear option (15+ generations stagnant)
            print(f"  💥 DIVERSITY BOOST LEVEL 3 (NUCLEAR):")
            print(f"     - Keeping only top 10% elite")
            print(f"     - Generating 50% new random models")
            print(f"     - Resetting temperature to {self.initial_temperature}")

            # Keep only top 10%
            sorted_pop = sorted(
                zip(self.population, self.elo_ratings),
                key=lambda x: x[1],
                reverse=True,
            )
            elite_count = max(2, int(self.population_size * 0.1))
            elites = [copy.deepcopy(model) for model, _ in sorted_pop[:elite_count]]

            # Generate 50% completely new models
            new_count = int(self.population_size * 0.5)
            new_models = []
            if self.architecture_enabled:
                # Equal distribution across architectures
                new_models = ArchitectureRegistry.create_population(
                    new_count, use_cuda=self.use_cuda
                )
            else:
                new_models = [
                    BasicEuchreNN(use_cuda=self.use_cuda) for _ in range(new_count)
                ]

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
