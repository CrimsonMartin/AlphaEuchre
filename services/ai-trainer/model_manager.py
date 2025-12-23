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
        Handles both old (130 features) and new (161 features) architectures.

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

            # Load the state dict to inspect it
            state_dict = torch.load(model_path)

            # Detect model architecture by checking first layer input size
            # Old models: 130 features, New models: 161 features
            first_layer_key = "card_network.0.weight"
            if first_layer_key in state_dict:
                input_size = state_dict[first_layer_key].shape[1]

                if input_size == 130:
                    # Old architecture
                    print(f"Loading old architecture model (130 features)")
                    model = BasicEuchreNN(
                        input_size=130,
                        card_hidden_sizes=[256, 128, 64],
                        trump_hidden_sizes=[128, 64],
                        discard_hidden_sizes=[64],
                    )
                elif input_size == 161:
                    # New architecture
                    print(f"Loading new architecture model (161 features)")
                    model = BasicEuchreNN(
                        input_size=161,
                        card_hidden_sizes=[512, 256, 128, 64],
                        trump_hidden_sizes=[256, 128, 64],
                        discard_hidden_sizes=[128, 64],
                    )
                else:
                    print(f"Unknown architecture with input size {input_size}")
                    return None
            else:
                # Fallback to new architecture
                print(f"Could not detect architecture, using new default")
                model = BasicEuchreNN()

            # Load weights
            model.load_state_dict(state_dict)
            model.eval()  # Set to evaluation mode

            return model

        except Exception as e:
            print(f"Error loading model {model_id}: {e}")
            import traceback

            traceback.print_exc()
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

    def get_current_best_model(self) -> Optional[Tuple[str, BasicEuchreNN, float]]:
        """
        Get the current best model across ALL training runs (for multi-instance training).

        Returns:
            Tuple of (model_id, model, elo_rating) or None if no models exist
        """
        try:
            conn = self.get_db_connection()
            cur = conn.cursor()

            # Get the model with highest ELO rating
            cur.execute(
                """
                SELECT id, model_path, elo_rating
                FROM ai_models
                WHERE active = true 
                  AND model_path IS NOT NULL
                  AND elo_rating IS NOT NULL
                ORDER BY elo_rating DESC
                LIMIT 1
                """,
            )

            row = cur.fetchone()
            cur.close()
            conn.close()

            if not row:
                return None

            model_id = row[0]
            model_path = row[1]
            elo_rating = row[2]

            # Load model
            if os.path.exists(model_path):
                model = BasicEuchreNN()
                model.load_state_dict(torch.load(model_path))
                model.eval()
                return (model_id, model, elo_rating)

            return None

        except Exception as e:
            print(f"Error getting current best model: {e}")
            return None

    def save_model_with_lock(
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
        Save a model with database locking to prevent race conditions in multi-instance training.
        Only saves if this model is better than the current best.

        Args:
            model: The neural network model to save
            name: Name for the model
            generation: Generation number
            fitness: Fitness score (deprecated, use elo_rating)
            training_run_id: ID of the training run
            is_best: Whether this is the best model from the run
            elo_rating: ELO rating of the model

        Returns:
            Model ID (UUID) or None if not saved
        """
        try:
            conn = self.get_db_connection()
            cur = conn.cursor()

            # Acquire advisory lock (lock ID: 12345)
            cur.execute("SELECT pg_advisory_lock(12345)")

            # Check current best ELO
            cur.execute(
                """
                SELECT MAX(elo_rating) FROM ai_models WHERE active = true
                """
            )
            current_best_elo = cur.fetchone()[0] or 0.0

            # Only save if this model is better
            if elo_rating > current_best_elo:
                model_id = str(uuid.uuid4())

                # Save model weights to filesystem
                model_filename = f"model_{model_id}.pt"
                model_path = os.path.join(self.models_dir, model_filename)
                torch.save(model.state_dict(), model_path)

                # Save metadata to database
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
                print(
                    f"✅ NEW GLOBAL BEST saved: {model_id} (ELO: {elo_rating:.0f}, previous: {current_best_elo:.0f})"
                )
                result = model_id
            else:
                print(
                    f"⏭️  Model not saved (ELO {elo_rating:.0f} <= current best {current_best_elo:.0f})"
                )
                result = None

            # Release advisory lock
            cur.execute("SELECT pg_advisory_unlock(12345)")

            cur.close()
            conn.close()

            return result

        except Exception as e:
            print(f"Error saving model with lock: {e}")
            return None
