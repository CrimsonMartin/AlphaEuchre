"""
AI Trainer Service - PyTorch Neural Network Training with Genetic Algorithm
"""

import os
import sys
import threading
import torch
import psycopg2
from flask import Flask, request, jsonify
from datetime import datetime
import uuid

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "shared"))

from genetic.genetic_algorithm import GeneticAlgorithm
from networks.basic_nn import BasicEuchreNN
from model_manager import ModelManager

app = Flask(__name__)

# Configuration
app.config["DATABASE_URL"] = os.getenv(
    "DATABASE_URL", "postgresql://euchre:euchre_dev_pass@postgres:5432/euchrebot"
)

# Global training state
training_runs = {}
training_lock = threading.Lock()


def get_db_connection():
    """Get database connection"""
    return psycopg2.connect(app.config["DATABASE_URL"])


def save_model_to_db(
    model: BasicEuchreNN,
    name: str,
    generation: int,
    fitness: float,
    training_run_id: str,
):
    """Save a trained model to the database"""
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        # Serialize model weights
        model_buffer = torch.save(model.state_dict(), f"/tmp/model_{uuid.uuid4()}.pt")

        # For now, save model path instead of BYTEA (simpler)
        model_id = str(uuid.uuid4())

        cur.execute(
            """
            INSERT INTO ai_models (id, name, version, architecture, generation, 
                                 training_run_id, performance_metrics, active)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """,
            (
                model_id,
                name,
                "v1.0",
                "BasicEuchreNN",
                generation,
                training_run_id,
                f'{{"fitness": {fitness}}}',
                True,
            ),
        )

        conn.commit()
        cur.close()
        conn.close()

        return model_id
    except Exception as e:
        print(f"Error saving model to DB: {e}")
        return None


def run_training(
    run_id: str, population_size: int, generations: int, use_seeding: bool = False
):
    """Run training in background thread"""
    try:
        with training_lock:
            training_runs[run_id]["status"] = "running"
            training_runs[run_id]["current_generation"] = 0

        # Create database entry for training run
        try:
            conn = get_db_connection()
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO training_runs (id, name, generation_count, population_size, 
                                         mutation_rate, crossover_rate, elite_size, status)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """,
                (
                    run_id,
                    f"Training Run {run_id[:8]}",
                    0,
                    population_size,
                    0.1,
                    0.7,
                    2,
                    "running",
                ),
            )
            conn.commit()
            cur.close()
            conn.close()
        except Exception as e:
            print(f"Error creating training run in DB: {e}")

        # Initialize model manager
        model_manager = ModelManager(app.config["DATABASE_URL"])

        # Get seed models if requested
        seed_models = None
        if use_seeding:
            print("Seeding population from best previous models (25%)")
            seed_models = model_manager.seed_population_from_best(
                population_size, seed_percentage=0.25
            )

        # Initialize genetic algorithm with 10 games per pairing
        ga = GeneticAlgorithm(
            population_size=population_size,
        )

        # Callback to update progress
        def progress_callback(gen, best_fitness, avg_fitness):
            with training_lock:
                training_runs[run_id]["current_generation"] = gen
                training_runs[run_id]["best_fitness"] = best_fitness
                training_runs[run_id]["avg_fitness"] = avg_fitness

            # Update database
            try:
                conn = get_db_connection()
                cur = conn.cursor()
                cur.execute(
                    """
                    UPDATE training_runs 
                    SET generation_count = %s
                    WHERE id = %s
                """,
                    (gen, run_id),
                )
                conn.commit()
                cur.close()
                conn.close()
            except Exception as e:
                print(f"Error updating training run: {e}")

        # Run evolution with seed models
        print(f"Starting training run {run_id}")
        best_model = ga.evolve(
            generations=generations, callback=progress_callback, seed_models=seed_models
        )

        # Save best model using ModelManager
        with training_lock:
            best_elo = training_runs[run_id].get(
                "best_fitness", 1500.0
            )  # Now stores ELO

        model_id = model_manager.save_model(
            best_model,
            f"BestModel-Gen{generations}",
            generations,
            best_elo,  # fitness parameter (for backwards compatibility)
            run_id,
            is_best=True,
            elo_rating=best_elo,
        )

        # Mark as complete
        with training_lock:
            training_runs[run_id]["status"] = "completed"
            training_runs[run_id]["best_model_id"] = model_id

        # Update database
        try:
            conn = get_db_connection()
            cur = conn.cursor()
            cur.execute(
                """
                UPDATE training_runs 
                SET status = %s, completed_at = %s
                WHERE id = %s
            """,
                ("completed", datetime.now(), run_id),
            )
            conn.commit()
            cur.close()
            conn.close()
        except Exception as e:
            print(f"Error completing training run: {e}")

        print(f"Training run {run_id} completed. Best model: {model_id}")

    except Exception as e:
        print(f"Training error: {e}")
        with training_lock:
            training_runs[run_id]["status"] = "failed"
            training_runs[run_id]["error"] = str(e)


@app.route("/health")
def health():
    return {"status": "healthy", "service": "ai-trainer"}


@app.route("/api/train/start", methods=["POST"])
def start_training():
    """Start a new genetic algorithm training run"""
    data = request.json
    population_size = data.get("population_size", 20)
    generations = data.get("generations", 10)
    use_seeding = data.get("use_seeding", False)

    # Create training run ID
    run_id = str(uuid.uuid4())

    # Initialize training state
    with training_lock:
        training_runs[run_id] = {
            "status": "starting",
            "population_size": population_size,
            "generations": generations,
            "use_seeding": use_seeding,
            "current_generation": 0,
            "best_fitness": 0.0,
            "avg_fitness": 0.0,
            "started_at": datetime.now().isoformat(),
        }

    # Start training in background thread
    thread = threading.Thread(
        target=run_training, args=(run_id, population_size, generations, use_seeding)
    )
    thread.daemon = True
    thread.start()

    return jsonify(
        {
            "status": "started",
            "training_run_id": run_id,
            "population_size": population_size,
            "generations": generations,
            "use_seeding": use_seeding,
        }
    )


@app.route("/api/train/status/<run_id>", methods=["GET"])
def training_status(run_id):
    """Get status of a training run"""
    with training_lock:
        if run_id in training_runs:
            return jsonify(training_runs[run_id])

    # Check database
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            """
            SELECT status, generation_count, started_at, completed_at
            FROM training_runs
            WHERE id = %s
        """,
            (run_id,),
        )
        row = cur.fetchone()
        cur.close()
        conn.close()

        if row:
            return jsonify(
                {
                    "status": row[0],
                    "current_generation": row[1],
                    "started_at": row[2].isoformat() if row[2] else None,
                    "completed_at": row[3].isoformat() if row[3] else None,
                }
            )
    except Exception as e:
        print(f"Error fetching training status: {e}")

    return jsonify({"error": "Training run not found"}), 404


@app.route("/api/models", methods=["GET"])
def list_models():
    """List all trained models"""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, name, architecture, generation, performance_metrics, created_at
            FROM ai_models
            WHERE active = true
            ORDER BY created_at DESC
            LIMIT 50
        """
        )
        rows = cur.fetchall()
        cur.close()
        conn.close()

        models = []
        for row in rows:
            fitness = 0.0
            if row[4]:  # performance_metrics
                import json

                metrics = json.loads(row[4])
                fitness = metrics.get("fitness", 0.0)

            models.append(
                {
                    "id": row[0],
                    "name": row[1],
                    "architecture": row[2],
                    "generation": row[3],
                    "fitness": fitness,
                    "created_at": row[5].isoformat() if row[5] else None,
                }
            )

        return jsonify({"models": models})
    except Exception as e:
        print(f"Error listing models: {e}")
        return jsonify({"models": []})


@app.route("/api/predict", methods=["POST"])
def predict():
    """Get move prediction from a model"""
    data = request.json
    game_state = data.get("game_state")
    model_id = data.get("model_id", "model-001")

    # This would load the model and make a prediction
    # For now, return a placeholder

    return jsonify(
        {
            "model_id": model_id,
            "recommended_card": "9H",
            "probabilities": {"9H": 0.85, "10H": 0.10, "JH": 0.05},
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002, debug=True)
