"""
ELO Rating System for AI Model Evaluation
"""

from typing import Dict, Tuple, List, Optional
import math


class ELORatingSystem:
    """
    ELO rating system for evaluating AI model performance.

    Standard ELO implementation with configurable K-factor.
    """

    def __init__(self, initial_rating: float = 1500, k_factor: float = 32):
        """
        Initialize ELO rating system.

        Args:
            initial_rating: Starting rating for new models (default: 1500)
            k_factor: Maximum rating change per game (default: 32)
        """
        self.initial_rating = initial_rating
        self.k_factor = k_factor
        self.ratings: Dict[int, float] = {}  # model_index -> rating

    def get_rating(self, model_index: int) -> float:
        """Get current rating for a model, or initial rating if not yet rated."""
        return self.ratings.get(model_index, self.initial_rating)

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """
        Calculate expected score for player A against player B.

        Args:
            rating_a: ELO rating of player A
            rating_b: ELO rating of player B

        Returns:
            Expected score (0.0 to 1.0) for player A
        """
        return 1.0 / (1.0 + math.pow(10, (rating_b - rating_a) / 400.0))

    def update_ratings(
        self, winner_index: int, loser_index: int, margin: int = 0
    ) -> Tuple[float, float]:
        """
        Update ratings after a match.

        Args:
            winner_index: Index of winning model
            loser_index: Index of losing model
            margin: Point margin (optional, for future use)

        Returns:
            Tuple of (new_winner_rating, new_loser_rating)
        """
        winner_rating = self.get_rating(winner_index)
        loser_rating = self.get_rating(loser_index)

        # Calculate expected scores
        expected_winner = self.expected_score(winner_rating, loser_rating)
        expected_loser = self.expected_score(loser_rating, winner_rating)

        # Update ratings (winner gets 1.0, loser gets 0.0)
        new_winner = winner_rating + self.k_factor * (1.0 - expected_winner)
        new_loser = loser_rating + self.k_factor * (0.0 - expected_loser)

        # Store updated ratings
        self.ratings[winner_index] = new_winner
        self.ratings[loser_index] = new_loser

        return new_winner, new_loser

    def update_team_ratings(
        self,
        team1_indices: List[int],
        team2_indices: List[int],
        team1_won: bool,
        team1_score: int,
        team2_score: int,
    ) -> Dict[int, float]:
        """
        Update ratings for team-based match.

        Uses average team rating for expected score calculation,
        then distributes rating changes to individual team members.

        Args:
            team1_indices: List of model indices on team 1
            team2_indices: List of model indices on team 2
            team1_won: True if team 1 won
            team1_score: Final score for team 1
            team2_score: Final score for team 2

        Returns:
            Dictionary of model_index -> new_rating for all participants
        """
        # Calculate average team ratings
        team1_avg = sum(self.get_rating(i) for i in team1_indices) / len(team1_indices)
        team2_avg = sum(self.get_rating(i) for i in team2_indices) / len(team2_indices)

        # Calculate expected scores
        expected_team1 = self.expected_score(team1_avg, team2_avg)
        expected_team2 = self.expected_score(team2_avg, team1_avg)

        # Actual scores (1 for win, 0 for loss)
        actual_team1 = 1.0 if team1_won else 0.0
        actual_team2 = 0.0 if team1_won else 1.0

        # Calculate rating changes
        team1_change = self.k_factor * (actual_team1 - expected_team1)
        team2_change = self.k_factor * (actual_team2 - expected_team2)

        # Apply changes to all team members
        updated_ratings = {}

        for idx in team1_indices:
            old_rating = self.get_rating(idx)
            new_rating = old_rating + team1_change
            self.ratings[idx] = new_rating
            updated_ratings[idx] = new_rating

        for idx in team2_indices:
            old_rating = self.get_rating(idx)
            new_rating = old_rating + team2_change
            self.ratings[idx] = new_rating
            updated_ratings[idx] = new_rating

        return updated_ratings

    def get_all_ratings(self) -> Dict[int, float]:
        """Get all current ratings."""
        return self.ratings.copy()

    def reset_ratings(self):
        """Reset all ratings to initial value."""
        self.ratings.clear()

    def get_rating_description(self, elo: float) -> str:
        """Get a textual description of the rating."""
        return f"Rating: {elo:.2f}"

    def update_individual_ratings(
        self,
        player_indices: List[int],
        tricks_won: Dict[int, int],
        trump_caller_index: Optional[int] = None,
        calling_team_won: bool = False,
    ) -> Dict[int, float]:
        """
        Update ratings based on individual trick performance.

        This replaces team-based win/loss with individual trick-based rewards.
        Models are rewarded for the number of tricks they personally won.

        Args:
            player_indices: List of 4 model indices in the hand
            tricks_won: Dictionary of player_index -> number of tricks won (0-5)
            trump_caller_index: Optional index of player who called trump
            calling_team_won: Whether the team that called trump won the hand (>=3 tricks)

        Returns:
            Dictionary of model_index -> new_rating for all participants
        """
        updated_ratings = {}

        # Calculate expected performance (should average 1.25 tricks per player)
        # but we'll use relative comparisons instead

        for idx in player_indices:
            old_rating = self.get_rating(idx)
            player_tricks = tricks_won.get(idx, 0)

            # Base rating change: proportional to tricks won
            # Scale from -K to +K based on 0-5 tricks
            # 0 tricks = big loss, 5 tricks = big gain, ~1-2 tricks = neutral
            normalized_performance = (
                player_tricks / 5.0
            ) * 2.0 - 0.5  # maps 0->-0.5, 5->1.5, 2.5->0.5
            base_change = self.k_factor * normalized_performance

            # Apply trump calling modifiers
            modifier = 1.0
            if trump_caller_index is not None and idx == trump_caller_index:
                if calling_team_won:
                    # Reward aggressive successful trump calls
                    modifier = 1.4  # 40% bonus
                else:
                    # Penalty for bad calls, but not too harsh to encourage trying
                    modifier = 0.6  # 40% penalty

            rating_change = base_change * modifier
            new_rating = old_rating + rating_change

            self.ratings[idx] = new_rating
            updated_ratings[idx] = new_rating

        return updated_ratings
