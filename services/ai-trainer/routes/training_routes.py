"""Training routes for AI Trainer service"""

import threading
import uuid
from datetime import datetime
from flask import Blueprint, jsonify, request, current_app

training_bp = Blueprint("training", __name__)

# Global training state (shared across app)
training_runs = {}
training_lock = threading.Lock()
cancellation_flags = {}


def get_training_state():
    """Get references to training state"""
    return training_runs, training_lock, cancellation_flags


@training_bp.route("/api/train/start", methods=["POST"])
def start_training():
    """Start a new continuous training run"""
    from training.trainer import run_continuous_training
    import psycopg2

    data = request.json
    population_size = data.get("population_size", 40)
    games_per_pairing = data.get("games_per_pairing", 100)

    # Create training run ID
    run_id = str(uuid.uuid4())

    # Initialize training state
    with training_lock:
        training_runs[run_id] = {
            "status": "starting",
            "population_size": population_size,
            "continuous": True,
            "current_generation": 0,
            "best_fitness": 1500.0,
            "avg_fitness": 1500.0,
            "started_at": datetime.now().isoformat(),
        }
        cancellation_flags[run_id] = False

    # Database connection function
    def get_db_connection():
        return psycopg2.connect(current_app.config["DATABASE_URL"])

    # Start training in background thread
    thread = threading.Thread(
        target=run_continuous_training,
        args=(
            run_id,
            population_size,
            games_per_pairing,
            current_app.config["DATABASE_URL"],
            current_app.config["NUM_WORKERS"],
            current_app.config["PARALLEL_MODE"],
            current_app.config["USE_CUDA"],
            training_runs,
            training_lock,
            cancellation_flags,
            get_db_connection,
        ),
    )
    thread.daemon = True
    thread.start()

    return jsonify(
        {
            "status": "started",
            "training_run_id": run_id,
            "population_size": population_size,
            "games_per_pairing": games_per_pairing,
            "continuous": True,
        }
    )


@training_bp.route("/api/train/cancel/<run_id>", methods=["POST"])
def cancel_training(run_id):
    """Cancel a running training session"""
    with training_lock:
        if run_id in cancellation_flags:
            cancellation_flags[run_id] = True
            return jsonify({"status": "cancelling", "run_id": run_id})
        else:
            return jsonify({"error": "Training run not found"}), 404


@training_bp.route("/api/train/status/<run_id>", methods=["GET"])
def training_status(run_id):
    """Get status of a training run"""
    import psycopg2

    with training_lock:
        if run_id in training_runs:
            return jsonify(training_runs[run_id])

    # Check database
    try:
        conn = psycopg2.connect(current_app.config["DATABASE_URL"])
        cur = conn.cursor()
        cur.execute(
            """
            SELECT status, generation_count, best_fitness, avg_fitness, started_at, completed_at
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
                    "best_fitness": row[2] or 1500.0,
                    "avg_fitness": row[3] or 1500.0,
                    "continuous": True,
                    "started_at": row[4].isoformat() if row[4] else None,
                    "completed_at": row[5].isoformat() if row[5] else None,
                }
            )
    except Exception as e:
        print(f"Error fetching training status: {e}")

    return jsonify({"error": "Training run not found"}), 404


@training_bp.route("/api/train/logs/<run_id>", methods=["GET"])
def training_logs(run_id):
    """Get training logs for a specific run"""
    import psycopg2

    try:
        conn = psycopg2.connect(current_app.config["DATABASE_URL"])
        cur = conn.cursor()
        cur.execute(
            """
            SELECT generation, best_fitness, avg_fitness, message, created_at
            FROM training_logs
            WHERE training_run_id = %s
            ORDER BY generation ASC
        """,
            (run_id,),
        )
        rows = cur.fetchall()
        cur.close()
        conn.close()

        logs = []
        for row in rows:
            logs.append(
                {
                    "generation": row[0],
                    "best_fitness": row[1],
                    "avg_fitness": row[2],
                    "message": row[3],
                    "created_at": row[4].isoformat() if row[4] else None,
                }
            )

        return jsonify({"logs": logs})
    except Exception as e:
        print(f"Error fetching training logs: {e}")
        return jsonify({"logs": []})
