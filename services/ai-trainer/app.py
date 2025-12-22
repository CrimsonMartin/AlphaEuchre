"""
AI Trainer Service - PyTorch Neural Network Training with Genetic Algorithm
Continuous Training Mode with ELO Ratings
"""

import os
import sys
import threading
import torch
import psycopg2
import copy
from flask import Flask, request, jsonify
from datetime import datetime
import uuid

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "shared"))

from genetic.genetic_algorithm import GeneticAlgorithm
from networks.basic_nn import BasicEuchreNN, encode_trump_state
from model_manager import ModelManager
import numpy as np

app = Flask(__name__)

# Configuration
app.config["DATABASE_URL"] = os.getenv(
    "DATABASE_URL", "postgresql://euchre:euchre_dev_pass@postgres:5432/euchrebot"
)

# Global training state
training_runs = {}
training_lock = threading.Lock()
cancellation_flags = {}  # Track which runs should be cancelled


def get_db_connection():
    """Get database connection"""
    return psycopg2.connect(app.config["DATABASE_URL"])


def run_continuous_training(
    run_id: str, population_size: int, games_per_pairing: int = 50
):
    """Run continuous training until cancelled"""
    try:
        with training_lock:
            training_runs[run_id]["status"] = "running"
            training_runs[run_id]["current_generation"] = 0
            cancellation_flags[run_id] = False

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
                    f"Continuous Training {run_id[:8]}",
                    0,
                    population_size,
                    0.1,
                    0.7,
                    3,
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

        # Always seed from best models (25%)
        print("Seeding population from best previous models (25%)")
        seed_models = model_manager.seed_population_from_best(
            population_size, seed_percentage=0.25
        )

        # Initialize genetic algorithm with enhanced parameters
        ga = GeneticAlgorithm(
            population_size=population_size,
            mutation_rate=0.15,
            crossover_rate=0.7,
            elite_size=4,
            games_per_pairing=games_per_pairing,
        )

        # Initialize population
        ga.initialize_population(seed_models)

        # Continuous training loop
        generation = 0
        while not cancellation_flags.get(run_id, False):
            generation += 1
            print(f"\n{'='*60}")
            print(f"Generation {generation} (Continuous Mode)")
            print(f"{'='*60}")

            # Evaluate population
            ga.elo_ratings = ga.evaluate_population_parallel()

            # Print individual model ELO ratings
            for i, elo in enumerate(ga.elo_ratings):
                desc = ga.elo_system.get_rating_description(elo)
                print(f"  Model {i+1}/{len(ga.population)}: ELO = {elo:.0f} ({desc})")

            # Sort by ELO rating
            sorted_pop = sorted(
                zip(ga.population, ga.elo_ratings),
                key=lambda x: x[1],
                reverse=True,
            )

            current_best_model, current_best_elo = sorted_pop[0]
            avg_elo = sum(ga.elo_ratings) / len(ga.elo_ratings)

            # Update global champion if current best is better
            if current_best_elo > ga.global_best_elo:
                ga.global_best_model = copy.deepcopy(current_best_model)
                ga.global_best_elo = current_best_elo
                print(f"\n🏆 NEW GLOBAL CHAMPION! ELO: {current_best_elo:.0f}")

            print(f"\nGeneration {generation} Summary:")
            print(f"  Current Best ELO:  {current_best_elo:.0f}")
            print(f"  Global Best ELO:   {ga.global_best_elo:.0f}")
            print(f"  Average ELO:       {avg_elo:.0f}")

            # Update training state
            with training_lock:
                training_runs[run_id]["current_generation"] = generation
                training_runs[run_id]["best_fitness"] = ga.global_best_elo
                training_runs[run_id]["avg_fitness"] = avg_elo

            # Update database with generation count and fitness values
            try:
                conn = get_db_connection()
                cur = conn.cursor()
                cur.execute(
                    """
                    UPDATE training_runs 
                    SET generation_count = %s, best_fitness = %s, avg_fitness = %s
                    WHERE id = %s
                """,
                    (generation, ga.global_best_elo, avg_elo, run_id),
                )

                # Insert training log entry
                cur.execute(
                    """
                    INSERT INTO training_logs (training_run_id, generation, best_fitness, avg_fitness, message)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (training_run_id, generation) DO NOTHING
                """,
                    (
                        run_id,
                        generation,
                        ga.global_best_elo,
                        avg_elo,
                        f"Generation {generation}: Best = {ga.global_best_elo:.0f}, Avg = {avg_elo:.0f}",
                    ),
                )

                conn.commit()
                cur.close()
                conn.close()
            except Exception as e:
                print(f"Error updating training run: {e}")

            # Auto-save best model every 5 generations (and at generation 1)
            if (
                generation == 1 or generation % 5 == 0
            ) and ga.global_best_model is not None:
                print(f"  💾 Auto-saving best model (generation {generation})...")
                model_manager.save_model(
                    ga.global_best_model,
                    f"AutoSave-Gen{generation}",
                    generation,
                    ga.global_best_elo,
                    run_id,
                    is_best=True,
                    elo_rating=ga.global_best_elo,
                )

            # Keep elite - ALWAYS include global champion first
            elites = []
            if ga.global_best_model is not None:
                elites.append(copy.deepcopy(ga.global_best_model))
                print(f"  ✓ Global champion preserved in population")

            # Add remaining elites from current generation
            for model, elo in sorted_pop[: ga.elite_size]:
                if len(elites) < ga.elite_size:
                    elites.append(copy.deepcopy(model))

            # Select parents
            parents = ga.selection()

            # Create offspring
            offspring = []
            for i in range(0, len(parents), 2):
                if i + 1 < len(parents):
                    child1 = ga.crossover(parents[i], parents[i + 1])
                    child2 = ga.crossover(parents[i + 1], parents[i])
                    ga.mutate(child1)
                    ga.mutate(child2)
                    offspring.extend([child1, child2])

            # New population
            ga.population = elites + offspring[: ga.population_size - ga.elite_size]

        # Training was cancelled
        print(f"\n⚠️  Training cancelled at generation {generation}")

        # Save final best model
        if ga.global_best_model is not None:
            print(f"  💾 Saving final best model...")
            model_id = model_manager.save_model(
                ga.global_best_model,
                f"FinalBest-Gen{generation}",
                generation,
                ga.global_best_elo,
                run_id,
                is_best=True,
                elo_rating=ga.global_best_elo,
            )

            with training_lock:
                training_runs[run_id]["best_model_id"] = model_id

        # Mark as completed
        with training_lock:
            training_runs[run_id]["status"] = "completed"

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

        print(f"Training run {run_id} completed.")

    except Exception as e:
        print(f"Training error: {e}")
        import traceback

        traceback.print_exc()
        with training_lock:
            training_runs[run_id]["status"] = "failed"
            training_runs[run_id]["error"] = str(e)


@app.route("/health")
def health():
    return {"status": "healthy", "service": "ai-trainer"}


@app.route("/api/train/start", methods=["POST"])
def start_training():
    """Start a new continuous training run"""
    data = request.json
    population_size = data.get("population_size", 40)
    games_per_pairing = data.get("games_per_pairing", 50)

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
        cancellation_flags[run_id] = False  # Initialize cancellation flag

    # Start training in background thread
    thread = threading.Thread(
        target=run_continuous_training,
        args=(run_id, population_size, games_per_pairing),
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


@app.route("/api/train/cancel/<run_id>", methods=["POST"])
def cancel_training(run_id):
    """Cancel a running training session"""
    with training_lock:
        if run_id in cancellation_flags:
            cancellation_flags[run_id] = True
            return jsonify({"status": "cancelling", "run_id": run_id})
        else:
            return jsonify({"error": "Training run not found"}), 404


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


@app.route("/api/train/logs/<run_id>", methods=["GET"])
def training_logs(run_id):
    """Get training logs for a specific run"""
    try:
        conn = get_db_connection()
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


@app.route("/api/models", methods=["GET"])
def list_models():
    """List all trained models"""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, name, architecture, generation, elo_rating, created_at
            FROM ai_models
            WHERE active = true
            ORDER BY elo_rating DESC
            LIMIT 50
        """
        )
        rows = cur.fetchall()
        cur.close()
        conn.close()

        models = []
        for row in rows:
            models.append(
                {
                    "id": row[0],
                    "name": row[1],
                    "architecture": row[2],
                    "generation": row[3],
                    "elo_rating": row[4] or 1500,
                    "created_at": row[5].isoformat() if row[5] else None,
                }
            )

        return jsonify({"models": models})
    except Exception as e:
        print(f"Error listing models: {e}")
        return jsonify({"models": []})


@app.route("/api/models/<model_id>/predict", methods=["POST"])
def predict_move(model_id):
    """Predict the best move for a given game state using a trained model"""
    try:
        data = request.json
        game_state = data.get("game_state")
        valid_cards = data.get("valid_cards", [])

        if not game_state:
            return jsonify({"error": "game_state required"}), 400

        # Load the model
        model_manager = ModelManager(app.config["DATABASE_URL"])
        model = model_manager.load_model(model_id)

        if not model:
            return jsonify({"error": "Model not found"}), 404

        # Encode game state
        from networks.basic_nn import encode_game_state

        state_encoding = encode_game_state(game_state)

        # Get prediction
        card_index = model.predict_card(state_encoding)

        # Map card index to actual card
        all_cards = []
        for suit in ["C", "D", "H", "S"]:
            for rank in ["9", "10", "J", "Q", "K", "A"]:
                all_cards.append(f"{rank}{suit}")

        # Get predicted card
        if 0 <= card_index < len(all_cards):
            predicted_card = all_cards[card_index]

            # If valid_cards provided, ensure prediction is valid
            if valid_cards and predicted_card not in valid_cards:
                # Fall back to first valid card
                predicted_card = valid_cards[0] if valid_cards else predicted_card

            return jsonify(
                {
                    "card": predicted_card,
                    "model_id": model_id,
                    "card_index": card_index,
                }
            )
        else:
            # Invalid index, return first valid card or error
            if valid_cards:
                return jsonify(
                    {
                        "card": valid_cards[0],
                        "model_id": model_id,
                        "fallback": True,
                    }
                )
            return jsonify({"error": "Invalid prediction"}), 500

    except Exception as e:
        print(f"Error predicting move: {e}")
        import traceback

        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/models/<model_id>/analyze-trump", methods=["GET"])
def analyze_trump_strategy(model_id):
    """
    Analyze a model's trump calling strategy by testing various scenarios.
    Returns detailed statistics on when the model calls vs passes.
    """
    try:
        # Load the model
        model_manager = ModelManager(app.config["DATABASE_URL"])
        model = model_manager.load_model(model_id)

        if not model:
            return jsonify({"error": "Model not found"}), 404

        # Card encoding reference
        all_cards = []
        for suit in ["C", "D", "H", "S"]:
            for rank in ["9", "10", "J", "Q", "K", "A"]:
                all_cards.append(f"{rank}{suit}")

        # Define test scenarios
        # Each scenario: (hand_cards, turned_up_card, description, category)
        scenarios = []

        # Suits for testing
        suits = ["C", "D", "H", "S"]
        suit_names = {"C": "Clubs", "D": "Diamonds", "H": "Hearts", "S": "Spades"}
        opposite_suit = {"C": "S", "S": "C", "D": "H", "H": "D"}

        for trump_suit in suits:
            left_suit = opposite_suit[trump_suit]

            # Strong hands - Both bowers
            scenarios.append(
                {
                    "hand": [
                        f"J{trump_suit}",
                        f"J{left_suit}",
                        f"A{trump_suit}",
                        f"K{trump_suit}",
                        f"Q{trump_suit}",
                    ],
                    "turned_up": f"9{trump_suit}",
                    "description": f"Both bowers + A,K,Q of {suit_names[trump_suit]}",
                    "category": "both_bowers",
                    "trump_count": 5,
                    "has_right": True,
                    "has_left": True,
                    "off_aces": 0,
                }
            )

            # Right bower + support
            scenarios.append(
                {
                    "hand": [
                        f"J{trump_suit}",
                        f"A{trump_suit}",
                        f"K{trump_suit}",
                        f"9{left_suit}",
                        f"10{left_suit}",
                    ],
                    "turned_up": f"Q{trump_suit}",
                    "description": f"Right bower + A,K of {suit_names[trump_suit]}",
                    "category": "right_bower",
                    "trump_count": 3,
                    "has_right": True,
                    "has_left": False,
                    "off_aces": 0,
                }
            )

            # Right bower alone
            scenarios.append(
                {
                    "hand": [
                        f"J{trump_suit}",
                        f"9{left_suit}",
                        f"10{left_suit}",
                        f"Q{left_suit}",
                        f"K{left_suit}",
                    ],
                    "turned_up": f"A{trump_suit}",
                    "description": f"Right bower only, no other trump",
                    "category": "right_bower_alone",
                    "trump_count": 1,
                    "has_right": True,
                    "has_left": False,
                    "off_aces": 0,
                }
            )

            # Left bower + support
            scenarios.append(
                {
                    "hand": [
                        f"J{left_suit}",
                        f"A{trump_suit}",
                        f"K{trump_suit}",
                        f"9{left_suit}",
                        f"10{left_suit}",
                    ],
                    "turned_up": f"Q{trump_suit}",
                    "description": f"Left bower + A,K of {suit_names[trump_suit]}",
                    "category": "left_bower",
                    "trump_count": 3,
                    "has_right": False,
                    "has_left": True,
                    "off_aces": 0,
                }
            )

            # Left bower alone
            scenarios.append(
                {
                    "hand": [
                        f"J{left_suit}",
                        f"9{trump_suit}",
                        f"10{left_suit}",
                        f"Q{left_suit}",
                        f"K{left_suit}",
                    ],
                    "turned_up": f"A{trump_suit}",
                    "description": f"Left bower + 9 of trump only",
                    "category": "left_bower_weak",
                    "trump_count": 2,
                    "has_right": False,
                    "has_left": True,
                    "off_aces": 0,
                }
            )

            # No bowers, strong trump
            scenarios.append(
                {
                    "hand": [
                        f"A{trump_suit}",
                        f"K{trump_suit}",
                        f"Q{trump_suit}",
                        f"10{trump_suit}",
                        f"9{left_suit}",
                    ],
                    "turned_up": f"9{trump_suit}",
                    "description": f"A,K,Q,10 of {suit_names[trump_suit]}, no bowers",
                    "category": "no_bowers_strong",
                    "trump_count": 4,
                    "has_right": False,
                    "has_left": False,
                    "off_aces": 0,
                }
            )

            # No bowers, medium trump
            scenarios.append(
                {
                    "hand": [
                        f"A{trump_suit}",
                        f"K{trump_suit}",
                        f"9{trump_suit}",
                        f"10{left_suit}",
                        f"Q{left_suit}",
                    ],
                    "turned_up": f"Q{trump_suit}",
                    "description": f"A,K,9 of {suit_names[trump_suit]}, no bowers",
                    "category": "no_bowers_medium",
                    "trump_count": 3,
                    "has_right": False,
                    "has_left": False,
                    "off_aces": 0,
                }
            )

            # Weak trump with off-aces
            other_suits = [s for s in suits if s != trump_suit and s != left_suit]
            scenarios.append(
                {
                    "hand": [
                        f"9{trump_suit}",
                        f"10{trump_suit}",
                        f"A{other_suits[0]}",
                        f"A{other_suits[1]}",
                        f"K{left_suit}",
                    ],
                    "turned_up": f"Q{trump_suit}",
                    "description": f"9,10 of trump + 2 off-aces",
                    "category": "weak_trump_off_aces",
                    "trump_count": 2,
                    "has_right": False,
                    "has_left": False,
                    "off_aces": 2,
                }
            )

            # Garbage hand
            scenarios.append(
                {
                    "hand": [
                        f"9{left_suit}",
                        f"10{left_suit}",
                        f"Q{other_suits[0]}",
                        f"K{other_suits[0]}",
                        f"9{other_suits[1]}",
                    ],
                    "turned_up": f"A{trump_suit}",
                    "description": f"No trump, no aces - garbage",
                    "category": "garbage",
                    "trump_count": 0,
                    "has_right": False,
                    "has_left": False,
                    "off_aces": 0,
                }
            )

            # Borderline - 2 trump no bowers
            scenarios.append(
                {
                    "hand": [
                        f"K{trump_suit}",
                        f"Q{trump_suit}",
                        f"A{left_suit}",
                        f"K{left_suit}",
                        f"Q{other_suits[0]}",
                    ],
                    "turned_up": f"9{trump_suit}",
                    "description": f"K,Q of trump + off-ace",
                    "category": "borderline",
                    "trump_count": 2,
                    "has_right": False,
                    "has_left": False,
                    "off_aces": 1,
                }
            )

        # Test each scenario at each position
        results = {
            "model_id": model_id,
            "total_scenarios": 0,
            "position_stats": {
                "0": {
                    "name": "1st (Left of Dealer)",
                    "calls": 0,
                    "passes": 0,
                    "total": 0,
                    "avg_confidence": 0,
                },
                "1": {
                    "name": "2nd",
                    "calls": 0,
                    "passes": 0,
                    "total": 0,
                    "avg_confidence": 0,
                },
                "2": {
                    "name": "3rd",
                    "calls": 0,
                    "passes": 0,
                    "total": 0,
                    "avg_confidence": 0,
                },
                "3": {
                    "name": "Dealer",
                    "calls": 0,
                    "passes": 0,
                    "total": 0,
                    "avg_confidence": 0,
                },
            },
            "category_stats": {},
            "bower_stats": {
                "both_bowers": {"calls": 0, "passes": 0, "total": 0},
                "right_only": {"calls": 0, "passes": 0, "total": 0},
                "left_only": {"calls": 0, "passes": 0, "total": 0},
                "no_bowers": {"calls": 0, "passes": 0, "total": 0},
            },
            "trump_count_stats": {},
            "detailed_results": [],
        }

        # Initialize category stats
        for scenario in scenarios:
            cat = scenario["category"]
            if cat not in results["category_stats"]:
                results["category_stats"][cat] = {
                    "calls": 0,
                    "passes": 0,
                    "total": 0,
                    "description": "",
                }

        # Initialize trump count stats
        for i in range(6):
            results["trump_count_stats"][str(i)] = {"calls": 0, "passes": 0, "total": 0}

        # Run each scenario at each position
        confidence_sums = {"0": 0, "1": 0, "2": 0, "3": 0}

        for scenario in scenarios:
            for position in range(4):
                # Create game state for encoding
                game_state = {
                    "hand": scenario["hand"],
                    "current_player_position": position,
                    "dealer_position": 3,  # Dealer is always position 3 for consistency
                }

                # Encode trump state
                trump_encoding = encode_trump_state(game_state, scenario["turned_up"])

                # Get model prediction with probabilities
                with torch.no_grad():
                    x = torch.FloatTensor(trump_encoding).unsqueeze(0).to(model.device)
                    output = model.forward_trump(x)
                    probs = output.cpu().numpy()[0]
                    decision_idx = int(np.argmax(probs))
                    confidence = float(probs[decision_idx])

                # decision_idx: 0-3 = call suits, 4 = pass
                is_call = decision_idx != 4
                called_suit = ["C", "D", "H", "S", "PASS"][decision_idx]

                # Update statistics
                results["total_scenarios"] += 1
                pos_key = str(position)

                if is_call:
                    results["position_stats"][pos_key]["calls"] += 1
                else:
                    results["position_stats"][pos_key]["passes"] += 1
                results["position_stats"][pos_key]["total"] += 1
                confidence_sums[pos_key] += confidence

                # Category stats
                cat = scenario["category"]
                if is_call:
                    results["category_stats"][cat]["calls"] += 1
                else:
                    results["category_stats"][cat]["passes"] += 1
                results["category_stats"][cat]["total"] += 1
                results["category_stats"][cat]["description"] = scenario["description"]

                # Bower stats
                if scenario["has_right"] and scenario["has_left"]:
                    bower_key = "both_bowers"
                elif scenario["has_right"]:
                    bower_key = "right_only"
                elif scenario["has_left"]:
                    bower_key = "left_only"
                else:
                    bower_key = "no_bowers"

                if is_call:
                    results["bower_stats"][bower_key]["calls"] += 1
                else:
                    results["bower_stats"][bower_key]["passes"] += 1
                results["bower_stats"][bower_key]["total"] += 1

                # Trump count stats
                tc_key = str(scenario["trump_count"])
                if is_call:
                    results["trump_count_stats"][tc_key]["calls"] += 1
                else:
                    results["trump_count_stats"][tc_key]["passes"] += 1
                results["trump_count_stats"][tc_key]["total"] += 1

                # Detailed result
                results["detailed_results"].append(
                    {
                        "hand": scenario["hand"],
                        "turned_up": scenario["turned_up"],
                        "position": position,
                        "position_name": results["position_stats"][pos_key]["name"],
                        "decision": "CALL" if is_call else "PASS",
                        "called_suit": called_suit,
                        "confidence": round(confidence, 3),
                        "category": cat,
                        "description": scenario["description"],
                        "trump_count": scenario["trump_count"],
                        "has_right": scenario["has_right"],
                        "has_left": scenario["has_left"],
                    }
                )

        # Calculate averages and percentages
        for pos_key in results["position_stats"]:
            stats = results["position_stats"][pos_key]
            if stats["total"] > 0:
                stats["call_rate"] = round(stats["calls"] / stats["total"] * 100, 1)
                stats["avg_confidence"] = round(
                    confidence_sums[pos_key] / stats["total"], 3
                )

        for cat in results["category_stats"]:
            stats = results["category_stats"][cat]
            if stats["total"] > 0:
                stats["call_rate"] = round(stats["calls"] / stats["total"] * 100, 1)

        for bower_key in results["bower_stats"]:
            stats = results["bower_stats"][bower_key]
            if stats["total"] > 0:
                stats["call_rate"] = round(stats["calls"] / stats["total"] * 100, 1)

        for tc_key in results["trump_count_stats"]:
            stats = results["trump_count_stats"][tc_key]
            if stats["total"] > 0:
                stats["call_rate"] = round(stats["calls"] / stats["total"] * 100, 1)

        return jsonify(results)

    except Exception as e:
        print(f"Error analyzing trump strategy: {e}")
        import traceback

        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002, debug=True)
