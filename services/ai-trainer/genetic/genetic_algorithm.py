"""
Genetic Algorithm for Evolving Euchre Neural Networks
Round-Robin Tournament System with Team-Based Evaluation
ELO Rating System with Parallel Processing
"""

import random
import torch
import copy
import sys
import os
from typing import List, Tuple, Dict
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# Add parent directory to path to import euchre_core
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "shared"))

from euchre_core.game import EuchreGame, GamePhase
from euchre_core.player import PlayerType
from euchre_core.card import Card, Suit
from networks.basic_nn import BasicEuchreNN, encode_game_state
from elo_system import ELORatingSystem


# Helper function for parallel processing (must be at module level)
def run_tournament_group_worker(args):
    """Worker function to run a tournament group in parallel"""
    models_state_dicts, games_per_pairing, all_cards = args

    # Reconstruct models from state dicts
    models = []
    for state_dict in models_state_dicts:
        model = BasicEuchreNN(use_cuda=False)  # Use CPU in workers
        model.load_state_dict(state_dict)
        model.eval()
        models.append(model)

    # Run tournament
    ga_temp = GeneticAlgorithm(games_per_pairing=games_per_pairing)
    ga_temp.all_cards = all_cards
    results = ga_temp.run_round_robin_tournament_elo(models)

    return results


class GeneticAlgorithm:
    """Genetic algorithm for evolving neural network weights with round-robin tournaments and ELO ratings"""

    def __init__(
        self,
        population_size=20,
        mutation_rate=0.1,
        crossover_rate=0.7,
        elite_size=3,
        games_per_pairing=10,
        num_workers=None,
        use_elo=True,
    ):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
        self.games_per_pairing = games_per_pairing
        self.num_workers = num_workers or max(1, multiprocessing.cpu_count() - 1)
        self.use_elo = use_elo
        self.population: List[BasicEuchreNN] = []
        self.elo_ratings: List[float] = []

        # ELO rating system
        self.elo_system = ELORatingSystem(initial_rating=1500, k_factor=32)

        # Global champion tracking (best model across all generations)
        self.global_best_model: BasicEuchreNN | None = None
        self.global_best_elo: float = 0.0

        # Card mapping for predictions
        self.all_cards = []
        for suit in ["C", "D", "H", "S"]:
            for rank in ["9", "10", "J", "Q", "K", "A"]:
                self.all_cards.append(f"{rank}{suit}")

    def initialize_population(self, seed_models: List[BasicEuchreNN] | None = None):
        """
        Create initial population, optionally seeded with existing models.

        Args:
            seed_models: Optional list of models to seed the population
        """
        self.population = []

        if seed_models:
            # Add seed models
            for model in seed_models:
                self.population.append(copy.deepcopy(model))

        # Fill remaining slots with new random models
        while len(self.population) < self.population_size:
            self.population.append(BasicEuchreNN())

        # Initialize ELO ratings
        self.elo_system.reset_ratings()
        self.elo_ratings = [self.elo_system.initial_rating] * len(self.population)

        print(f"Initialized population of {len(self.population)} models")
        print(f"Using {self.num_workers} parallel workers")
        print(f"CUDA available: {torch.cuda.is_available()}")

    def play_game_with_teams(
        self,
        team1_models: Tuple[BasicEuchreNN, BasicEuchreNN],
        team2_models: Tuple[BasicEuchreNN, BasicEuchreNN],
    ) -> Tuple[int, int, int]:
        """
        Play a single game with specified teams.

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
                # Simple strategy: pass for now (can be improved)
                try:
                    game.pass_trump()
                except:
                    # Dealer must call
                    if game.state.turned_up_card:
                        game.call_trump(game.state.turned_up_card.suit)

            elif game.state.phase == GamePhase.TRUMP_SELECTION_ROUND2:
                # Simple strategy: call a random suit (not turned up)
                turned_up_suit = (
                    game.state.turned_up_card.suit
                    if game.state.turned_up_card
                    else Suit.CLUBS
                )
                available_suits = [
                    s
                    for s in [Suit.CLUBS, Suit.DIAMONDS, Suit.HEARTS, Suit.SPADES]
                    if s != turned_up_suit
                ]
                if game.state.current_player_position == game.state.dealer_position:
                    # Dealer must call
                    game.call_trump(random.choice(available_suits))
                else:
                    try:
                        game.pass_trump()
                    except:
                        game.call_trump(random.choice(available_suits))

            elif game.state.phase == GamePhase.DEALER_DISCARD:
                # Dealer discards lowest card
                dealer = game.state.get_player(game.state.dealer_position)
                card_to_discard = dealer.hand[0]  # Simple: discard first card
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
            fill_models = random.sample(self.population[:-remaining], 4 - remaining)
            group_models = remaining_models + fill_models
            group_indices = list(
                range(len(self.population) - remaining, len(self.population))
            )
            groups.append((group_models, group_indices))

        print(f"Running {len(groups)} tournament groups in parallel...")

        # Run tournaments for each group
        for group_idx, (group_models, group_indices) in enumerate(groups):
            print(f"Tournament Group {group_idx + 1}/{len(groups)}")
            self.run_round_robin_tournament_elo(group_models, group_indices)

        # Get final ELO ratings
        elo_ratings = [
            self.elo_system.get_rating(i) for i in range(len(self.population))
        ]

        return elo_ratings

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
        """Mutate model weights"""
        for param in model.parameters():
            if random.random() < self.mutation_rate:
                # Add Gaussian noise to weights
                noise = torch.randn_like(param) * 0.1
                param.data += noise

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
            if current_best_elo > self.global_best_elo:
                self.global_best_model = copy.deepcopy(current_best_model)
                self.global_best_elo = current_best_elo
                print(f"\n🏆 NEW GLOBAL CHAMPION! ELO: {current_best_elo:.0f}")

            print(f"\nGeneration {generation + 1} Summary:")
            print(f"  Current Best ELO:  {current_best_elo:.0f}")
            print(f"  Global Best ELO:   {self.global_best_elo:.0f}")
            print(f"  Average ELO:       {avg_elo:.0f}")

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
