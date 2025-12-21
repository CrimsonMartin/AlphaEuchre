"""
Main web UI routes
"""

from flask import (
    Blueprint,
    render_template,
    session,
    request,
    redirect,
    url_for,
    current_app,
)
import requests

main_bp = Blueprint("main", __name__)


@main_bp.route("/")
def index():
    """Home page"""
    return render_template("index.html")


@main_bp.route("/new-game", methods=["GET", "POST"])
def new_game():
    """Create a new game"""
    if request.method == "POST":
        # Get player names from form
        players = [
            {"name": request.form.get("player0", "Player 1"), "type": "human"},
            {
                "name": request.form.get("player1", "Player 2"),
                "type": request.form.get("type1", "random_ai"),
            },
            {
                "name": request.form.get("player2", "Player 3"),
                "type": request.form.get("type2", "random_ai"),
            },
            {
                "name": request.form.get("player3", "Player 4"),
                "type": request.form.get("type3", "random_ai"),
            },
        ]

        # Create game via API
        api_url = current_app.config["EUCHRE_API_URL"]
        try:
            response = requests.post(
                f"{api_url}/api/games", json={"players": players}, timeout=5
            )

            if response.status_code == 201:
                data = response.json()
                session["game_id"] = data["game_id"]
                session["player_position"] = 0
                return redirect(url_for("main.play_game"))
        except Exception as e:
            return render_template("error.html", error=str(e))

    return render_template("new_game.html")


@main_bp.route("/game")
def play_game():
    """Play game interface"""
    game_id = session.get("game_id")
    if not game_id:
        return redirect(url_for("main.index"))

    player_position = session.get("player_position", 0)

    # Get game state from API
    api_url = current_app.config["EUCHRE_API_URL"]
    try:
        response = requests.get(
            f"{api_url}/api/games/{game_id}",
            params={"perspective": player_position},
            timeout=5,
        )

        if response.status_code == 200:
            game_state = response.json()
            return render_template(
                "game.html", game_state=game_state, player_position=player_position
            )
    except Exception as e:
        return render_template("error.html", error=str(e))

    return render_template("game.html", game_state={}, player_position=player_position)


@main_bp.route("/history")
def history():
    """View game history"""
    return render_template("history.html")


@main_bp.route("/training", methods=["GET", "POST"])
def training():
    """AI training dashboard"""
    # Get training_run_id from session (persists across page refreshes)
    training_run_id = session.get("training_run_id")

    if request.method == "POST":
        population_size = request.form.get("population_size", 20, type=int)

        # Call the AI trainer API to start training
        api_url = current_app.config.get("AI_TRAINER_URL", "http://ai-trainer:5003")

        current_app.logger.info(f"=== CONTINUOUS TRAINING REQUEST ===")
        current_app.logger.info(f"API URL: {api_url}")
        current_app.logger.info(f"Population Size: {population_size}")
        current_app.logger.info(f"Mode: Continuous (infinite)")
        current_app.logger.info(f"Seeding: Always enabled (25%)")

        try:
            full_url = f"{api_url}/api/train/start"
            payload = {
                "population_size": population_size,
                "continuous": True,  # Always continuous
            }

            current_app.logger.info(f"Sending POST to: {full_url}")
            current_app.logger.info(f"Payload: {payload}")

            response = requests.post(
                full_url,
                json=payload,
                timeout=10,
            )

            current_app.logger.info(f"Response Status: {response.status_code}")
            current_app.logger.info(f"Response Body: {response.text}")

            if response.status_code == 200:
                current_app.logger.info("Training started successfully!")
                data = response.json()
                training_run_id = data.get("training_run_id")
                # Store in session so it persists across page refreshes
                session["training_run_id"] = training_run_id
            else:
                current_app.logger.error(
                    f"Training failed with status {response.status_code}"
                )
        except requests.exceptions.ConnectionError as e:
            current_app.logger.error(f"Connection error to AI trainer: {e}")
            current_app.logger.error(f"Could not connect to {api_url}")
        except requests.exceptions.Timeout as e:
            current_app.logger.error(f"Timeout connecting to AI trainer: {e}")
        except Exception as e:
            current_app.logger.error(f"Unexpected error starting training: {e}")
            current_app.logger.exception("Full traceback:")

    return render_template("training.html", training_run_id=training_run_id)


@main_bp.route("/game/move", methods=["POST"])
def play_card():
    """Play a card in the current game"""
    game_id = session.get("game_id")
    if not game_id:
        return redirect(url_for("main.index"))

    card = request.form.get("card")
    api_url = current_app.config["EUCHRE_API_URL"]

    try:
        response = requests.post(
            f"{api_url}/api/games/{game_id}/move", json={"card": card}, timeout=5
        )

        if response.status_code == 200:
            return redirect(url_for("main.play_game"))
        else:
            error = response.json().get("error", "Failed to play card")
            return render_template("error.html", error=error)
    except Exception as e:
        return render_template("error.html", error=str(e))


@main_bp.route("/game/trump", methods=["POST"])
def call_trump_action():
    """Call trump or pass"""
    game_id = session.get("game_id")
    if not game_id:
        return redirect(url_for("main.index"))

    suit = request.form.get("suit")
    pass_trump = request.form.get("pass") == "true"
    go_alone = request.form.get("go_alone") == "true"

    api_url = current_app.config["EUCHRE_API_URL"]

    try:
        response = requests.post(
            f"{api_url}/api/games/{game_id}/trump",
            json={"suit": suit, "pass": pass_trump, "go_alone": go_alone},
            timeout=5,
        )

        if response.status_code == 200:
            return redirect(url_for("main.play_game"))
        else:
            error = response.json().get("error", "Failed to call trump")
            return render_template("error.html", error=error)
    except Exception as e:
        return render_template("error.html", error=str(e))


@main_bp.route("/game/discard", methods=["POST"])
def discard_card():
    """Dealer discards a card after picking up in round 1"""
    game_id = session.get("game_id")
    if not game_id:
        return redirect(url_for("main.index"))

    card = request.form.get("card")
    api_url = current_app.config["EUCHRE_API_URL"]

    try:
        response = requests.post(
            f"{api_url}/api/games/{game_id}/discard", json={"card": card}, timeout=5
        )

        if response.status_code == 200:
            return redirect(url_for("main.play_game"))
        else:
            error = response.json().get("error", "Failed to discard card")
            return render_template("error.html", error=error)
    except Exception as e:
        return render_template("error.html", error=str(e))


@main_bp.route("/game/new-hand", methods=["POST"])
def start_new_hand():
    """Start a new hand after the current hand is complete"""
    game_id = session.get("game_id")
    if not game_id:
        return redirect(url_for("main.index"))

    api_url = current_app.config["EUCHRE_API_URL"]

    try:
        response = requests.post(f"{api_url}/api/games/{game_id}/new-hand", timeout=5)

        if response.status_code == 200:
            return redirect(url_for("main.play_game"))
        else:
            error = response.json().get("error", "Failed to start new hand")
            return render_template("error.html", error=error)
    except Exception as e:
        return render_template("error.html", error=str(e))


@main_bp.route("/api/games/<game_id>/valid-moves", methods=["GET"])
def get_valid_moves(game_id):
    """Proxy endpoint for getting valid moves from the euchre-api service"""
    from flask import jsonify

    perspective = request.args.get("perspective", type=int)
    api_url = current_app.config["EUCHRE_API_URL"]

    try:
        params = {}
        if perspective is not None:
            params["perspective"] = perspective

        response = requests.get(
            f"{api_url}/api/games/{game_id}/valid-moves",
            params=params,
            timeout=5,
        )

        if response.status_code == 200:
            return jsonify(response.json())
        else:
            return jsonify({"error": "Failed to get valid moves"}), response.status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@main_bp.route("/api/train/status/<run_id>", methods=["GET"])
def get_training_status(run_id):
    """Proxy endpoint for getting training status from ai-trainer service"""
    from flask import jsonify

    api_url = current_app.config.get("AI_TRAINER_URL", "http://ai-trainer:5003")

    try:
        response = requests.get(
            f"{api_url}/api/train/status/{run_id}",
            timeout=5,
        )

        if response.status_code == 200:
            return jsonify(response.json())
        else:
            return jsonify({"error": "Training run not found"}), response.status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@main_bp.route("/api/train/cancel/<run_id>", methods=["POST"])
def cancel_training(run_id):
    """Proxy endpoint for cancelling training from ai-trainer service"""
    from flask import jsonify

    api_url = current_app.config.get("AI_TRAINER_URL", "http://ai-trainer:5003")

    try:
        response = requests.post(
            f"{api_url}/api/train/cancel/{run_id}",
            timeout=5,
        )

        if response.status_code == 200:
            # Clear from session when cancelled
            if session.get("training_run_id") == run_id:
                session.pop("training_run_id", None)
            return jsonify(response.json())
        else:
            return jsonify({"error": "Training run not found"}), response.status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 500
