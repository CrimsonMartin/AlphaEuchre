"""
Model Manager - Handles saving, loading, and managing neural network models
"""

import os
import torch
import psycopg2
import uuid
from typing import List, Optional, Tuple
from datetime import datetime
from networks.basic_nn import BasicEuchreNN


class ModelManager:
    """Manages AI model persistence and retrieval"""

    def __init__(self, db_url: str, models_dir: str = "/models"):
        self.db_url = db_url
        self.models_dir = models_dir

        # Create models directory if it doesn't exist
        os.makedirs(self.models_dir, exist_ok=True)

    def get_db_connection(self):
        """Get database connection"""
        return psycopg2.connect(self.db_url)

    def save_model(
        self,
        model: BasicEuchreNN,
        name: str,
        generation: int,
        fitness: float,
        training_run_id: str,
        is_best: bool = False,
        elo_rating: float = 1500,
    ) -> str:
        """
        Save a trained model to filesystem and database.

        Args:
            model: The neural network model to save
            name: Name for the model
            generation: Generation number
            fitness: Fitness score (deprecated, use elo_rating)
            training_run_id: ID of the training run
            is_best: Whether this is the best model from the run
            elo_rating: ELO rating of the model

        Returns:
            Model ID (UUID)
        """
        try:
            model_id = str(uuid.uuid4())

            # Save model weights to filesystem
            model_filename = f"model_{model_id}.pt"
            model_path = os.path.join(self.models_dir, model_filename)
            torch.save(model.state_dict(), model_path)

            # Save metadata to database
            conn = self.get_db_connection()
            cur = conn.cursor()

            cur.execute(
                """
                INSERT INTO ai_models (
                    id, name, version, architecture, generation, 
                    training_run_id, performance_metrics, model_path, 
                    active, is_best_overall, elo_rating
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    model_id,
                    name,
                    "v1.0",
                    "BasicEuchreNN",
                    generation,
                    training_run_id,
                    f'{{"fitness": {fitness}, "elo_rating": {elo_rating}}}',
                    model_path,
                    True,
                    is_best,
                    elo_rating,
                ),
            )

            conn.commit()
            cur.close()
            conn.close()

            print(f"Model saved: {model_id} at {model_path} (ELO: {elo_rating:.0f})")
            return model_id

        except Exception as e:
            print(f"Error saving model: {e}")
            return None

    def load_model(self, model_id: str) -> Optional[BasicEuchreNN]:
        """
        Load a model from filesystem by ID.

        Args:
            model_id: UUID of the model to load

        Returns:
            Loaded BasicEuchreNN model or None if not found
        """
        try:
            conn = self.get_db_connection()
            cur = conn.cursor()

            cur.execute(
                """
                SELECT model_path, architecture
                FROM ai_models
                WHERE id = %s AND active = true
                """,
                (model_id,),
            )

            row = cur.fetchone()
            cur.close()
            conn.close()

            if not row:
                print(f"Model {model_id} not found in database")
                return None

            model_path = row[0]

            if not os.path.exists(model_path):
                print(f"Model file not found: {model_path}")
                return None

            # Create model instance and load weights
            model = BasicEuchreNN()
            model.load_state_dict(torch.load(model_path))
            model.eval()  # Set to evaluation mode

            return model

        except Exception as e:
            print(f"Error loading model {model_id}: {e}")
            return None

    def get_best_models(self, n: int = 5) -> List[Tuple[str, BasicEuchreNN, float]]:
        """
        Get the top N best models from all training runs.

        Args:
            n: Number of best models to retrieve

        Returns:
            List of tuples (model_id, model, fitness_score)
        """
        try:
            conn = self.get_db_connection()
            cur = conn.cursor()

            # Query for best models by fitness score
            cur.execute(
                """
                SELECT id, model_path, performance_metrics
                FROM ai_models
                WHERE active = true 
                  AND model_path IS NOT NULL
                  AND performance_metrics IS NOT NULL
                ORDER BY (performance_metrics->>'fitness')::float DESC
                LIMIT %s
                """,
                (n,),
            )

            rows = cur.fetchall()
            cur.close()
            conn.close()

            best_models = []
            for row in rows:
                model_id = row[0]
                model_path = row[1]

                # Extract fitness from JSON
                import json

                metrics = json.loads(row[2])
                fitness = metrics.get("fitness", 0.0)

                # Load model
                if os.path.exists(model_path):
                    model = BasicEuchreNN()
                    model.load_state_dict(torch.load(model_path))
                    model.eval()
                    best_models.append((model_id, model, fitness))

            print(f"Loaded {len(best_models)} best models")
            return best_models

        except Exception as e:
            print(f"Error getting best models: {e}")
            return []

    def seed_population_from_best(
        self, population_size: int, seed_percentage: float = 0.25
    ) -> List[BasicEuchreNN]:
        """
        Create a new population seeded with best models from previous runs.

        Args:
            population_size: Total size of population to create
            seed_percentage: Percentage of population to seed (0.0 to 1.0)

        Returns:
            List of BasicEuchreNN models
        """
        seed_count = int(population_size * seed_percentage)
        new_count = population_size - seed_count

        population = []

        # Get best models for seeding
        best_models = self.get_best_models(n=seed_count)

        if best_models:
            print(f"Seeding {len(best_models)} models from previous runs")
            for model_id, model, fitness in best_models:
                population.append(model)
                print(f"  Seeded model {model_id[:8]} with fitness {fitness:.3f}")

        # Fill remaining slots with new random models
        remaining = population_size - len(population)
        print(f"Creating {remaining} new random models")
        for _ in range(remaining):
            population.append(BasicEuchreNN())

        return population

    def update_model_stats(
        self, model_id: str, wins: int, losses: int, games_played: int
    ):
        """
        Update model statistics after games.

        Args:
            model_id: UUID of the model
            wins: Number of wins to add
            losses: Number of losses to add
            games_played: Number of games played to add
        """
        try:
            conn = self.get_db_connection()
            cur = conn.cursor()

            cur.execute(
                """
                UPDATE ai_models
                SET wins = wins + %s,
                    losses = losses + %s,
                    games_played = games_played + %s
                WHERE id = %s
                """,
                (wins, losses, games_played, model_id),
            )

            conn.commit()
            cur.close()
            conn.close()

        except Exception as e:
            print(f"Error updating model stats: {e}")

    def mark_as_best(self, model_id: str):
        """
        Mark a model as one of the best overall.

        Args:
            model_id: UUID of the model
        """
        try:
            conn = self.get_db_connection()
            cur = conn.cursor()

            cur.execute(
                """
                UPDATE ai_models
                SET is_best_overall = true
                WHERE id = %s
                """,
                (model_id,),
            )

            conn.commit()
            cur.close()
            conn.close()

        except Exception as e:
            print(f"Error marking model as best: {e}")
