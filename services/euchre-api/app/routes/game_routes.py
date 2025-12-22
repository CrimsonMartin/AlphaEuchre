"""
Game management routes
"""

from flask import Blueprint, request, jsonify, current_app
import json
import uuid
import random
import requests
from euchre_core import EuchreGame, PlayerType, Card, Suit
from euchre_core.game import GamePhase
from app import redis_client

game_bp = Blueprint("game", __name__)


def get_neural_net_prediction(game_state, model_id, valid_cards):
    """Call ai-trainer service to get neural network prediction"""
    try:
        ai_trainer_url = current_app.config.get(
            "AI_TRAINER_URL", "http://ai-trainer:5003"
        )

        response = requests.post(
            f"{ai_trainer_url}/api/models/{model_id}/predict",
            json={
                "game_state": game_state,
                "valid_cards": [str(card) for card in valid_cards],
            },
            timeout=5,
        )

        if response.status_code == 200:
            data = response.json()
            return data.get("card")
        else:
            print(f"Neural net prediction failed: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error calling neural net prediction: {e}")
        return None


def process_ai_turns(game: EuchreGame, max_iterations=10):
    """
    Process AI player turns until it's a human player's turn or game phase changes.
    Returns True if any AI turns were processed.
    """
    iterations = 0
    ai_played = False

    while iterations < max_iterations:
        current_player = game.state.get_current_player()

        # Stop if current player is human
        if current_player.player_type == PlayerType.HUMAN:
            break

        # Stop if game is over or in setup
        if game.state.phase in [
            GamePhase.SETUP,
            GamePhase.GAME_OVER,
            GamePhase.HAND_COMPLETE,
        ]:
            break

        # AI player's turn - make a decision
        if game.state.phase in [
            GamePhase.TRUMP_SELECTION_ROUND1,
            GamePhase.TRUMP_SELECTION_ROUND2,
        ]:
            # Trump selection phase
            if game.state.phase == GamePhase.TRUMP_SELECTION_ROUND1:
                # Simple AI: 10% chance to call trump in round 1
                if random.random() < 0.1:
                    game.call_trump(suit=None, go_alone=False)
                    ai_played = True
                else:
                    game.pass_trump()
                    ai_played = True
            else:
                # Round 2: Must call if dealer, otherwise 70% chance
                is_dealer = (
                    game.state.current_player_position == game.state.dealer_position
                )
                if is_dealer or random.random() < 0.7:
                    # Pick a random suit (not the turned up card's suit)
                    turned_up_suit = (
                        game.state.turned_up_card.suit
                        if game.state.turned_up_card
                        else None
                    )
                    available_suits = [
                        s
                        for s in [Suit.HEARTS, Suit.DIAMONDS, Suit.CLUBS, Suit.SPADES]
                        if s != turned_up_suit
                    ]
                    suit = random.choice(available_suits)
                    game.call_trump(suit=suit, go_alone=False)
                    ai_played = True
                else:
                    game.pass_trump()
                    ai_played = True

        elif game.state.phase == GamePhase.DEALER_DISCARD:
            # Dealer must discard a card after picking up
            dealer = game.state.get_player(game.state.dealer_position)
            if dealer.hand:
                # Simple AI: discard a random card (could be smarter)
                card_to_discard = random.choice(dealer.hand)
                game.dealer_discard(card_to_discard)
                ai_played = True
            else:
                # No cards to discard - shouldn't happen
                break

        elif game.state.phase == GamePhase.PLAYING:
            # Playing phase - get card to play
            valid_cards = game.get_valid_moves(game.state.current_player_position)
            if valid_cards:
                card = None

                # Check if this is a neural net AI with a model ID
                if (
                    current_player.player_type == PlayerType.NEURAL_NET_AI
                    and current_player.ai_model_id
                ):
                    # Get game state for prediction
                    game_state = game.get_state(
                        perspective_position=game.state.current_player_position
                    )

                    # Call neural net for prediction
                    card_str = get_neural_net_prediction(
                        game_state, current_player.ai_model_id, valid_cards
                    )

                    if card_str:
                        try:
                            card = Card.from_string(card_str)
                            # Verify it's in valid cards
                            if card not in valid_cards:
                                card = None
                        except:
                            card = None

                # Fall back to random if neural net failed or not applicable
                if card is None:
                    card = random.choice(valid_cards)

                game.play_card(card)
                ai_played = True
            else:
                # No valid cards - shouldn't happen, but break to avoid infinite loop
                break
        else:
            # Unknown phase, break
            break

        iterations += 1

    return ai_played


def save_game_to_redis(game: EuchreGame):
    """Save game state to Redis"""
    key = f"game:{game.state.game_id}"
    state_dict = game.state.to_dict(include_hands=True)
    redis_client.set(key, json.dumps(state_dict), ex=3600)  # 1 hour expiry


def load_game_from_redis(game_id: str) -> EuchreGame:
    """Load game state from Redis"""
    key = f"game:{game_id}"
    data = redis_client.get(key)
    if not data:
        return None

    state_dict = json.loads(data)

    # Reconstruct the game from saved state
    game = EuchreGame(game_id)

    # Restore players
    for player_data in state_dict["players"]:
        player = game.add_player(
            player_data["name"], PlayerType[player_data["type"].upper()]
        )
        # Restore ai_model_id if present
        if "ai_model_id" in player_data and player_data["ai_model_id"]:
            player.ai_model_id = player_data["ai_model_id"]

    # Restore game state
    from euchre_core.game import GamePhase

    game.state.phase = GamePhase(state_dict["phase"])
    game.state.hand_number = state_dict["hand_number"]
    game.state.dealer_position = state_dict["dealer_position"]
    game.state.current_player_position = state_dict["current_player_position"]
    game.state.trump_caller_position = state_dict["trump_caller_position"]
    game.state.going_alone = state_dict["going_alone"]
    game.state.alone_player_position = state_dict["alone_player_position"]
    game.state.team1_score = state_dict["team1_score"]
    game.state.team2_score = state_dict["team2_score"]
    game.state.team1_tricks = state_dict["team1_tricks"]
    game.state.team2_tricks = state_dict["team2_tricks"]

    # Restore trump and turned up card
    if state_dict["trump"]:
        game.state.trump = Suit(state_dict["trump"])
    if state_dict["turned_up_card"]:
        game.state.turned_up_card = Card.from_string(state_dict["turned_up_card"])

    # Restore player hands
    for i, player_data in enumerate(state_dict["players"]):
        if "hand" in player_data:
            cards = [Card.from_string(card_str) for card_str in player_data["hand"]]
            game.state.players[i].hand = cards

    # Restore current trick if exists
    if state_dict["current_trick"]:
        from euchre_core.trick import Trick

        trick_data = state_dict["current_trick"]
        game.state.current_trick = Trick(trick_data["lead_position"], game.state.trump)
        for card_data in trick_data["cards"]:
            card = Card.from_string(card_data["card"])
            game.state.current_trick.cards.append((card_data["position"], card))

    # Restore tricks_played list
    if state_dict["tricks_played"]:
        from euchre_core.trick import Trick

        for trick_data in state_dict["tricks_played"]:
            trick = Trick(trick_data["lead_position"], game.state.trump)
            for card_data in trick_data["cards"]:
                card = Card.from_string(card_data["card"])
                trick.cards.append((card_data["position"], card))
            game.state.tricks_played.append(trick)

    return game


@game_bp.route("/games", methods=["POST"])
def create_game():
    """Create a new game"""
    data = request.json
    players = data.get("players", [])

    if len(players) != 4:
        return jsonify({"error": "Exactly 4 players required"}), 400

    game_id = str(uuid.uuid4())
    game = EuchreGame(game_id)

    for player_data in players:
        name = player_data.get("name", "Player")
        player_type_str = player_data.get("type", "human")

        # Parse neural_net_ai:model_id format
        model_id = None
        if player_type_str.startswith("neural_net_ai:"):
            player_type = PlayerType.NEURAL_NET_AI
            model_id = player_type_str.split(":", 1)[1]
        elif player_type_str == "random_ai":
            player_type = PlayerType.RANDOM_AI
        elif player_type_str == "neural_net_ai":
            player_type = PlayerType.NEURAL_NET_AI
        else:
            player_type = PlayerType.HUMAN

        player = game.add_player(name, player_type)

        # Store model ID if this is a neural net player
        if model_id:
            player.ai_model_id = model_id

    # Start first hand
    game.start_new_hand()

    save_game_to_redis(game)

    return jsonify({"game_id": game_id, "state": game.get_state()}), 201


@game_bp.route("/games/<game_id>", methods=["GET"])
def get_game(game_id):
    """Get game state"""
    perspective = request.args.get("perspective", type=int)

    game = load_game_from_redis(game_id)
    if not game:
        return jsonify({"error": "Game not found"}), 404

    # Process AI turns until it's a human player's turn
    ai_played = process_ai_turns(game)
    if ai_played:
        save_game_to_redis(game)

    state = game.get_state(perspective_position=perspective)

    # Format for web UI
    formatted_state = {
        "game_id": state["game_id"],
        "phase": state["phase"],
        "trump": state["trump"],
        "current_player": (
            state["players"][state["current_player_position"]]["name"]
            if state["players"]
            else "Unknown"
        ),
        "current_player_position": state["current_player_position"],
        "dealer_position": state["dealer_position"],
        "players": [p["name"] for p in state["players"]],
        "score": {"team1": state["team1_score"], "team2": state["team2_score"]},
        "hand": (
            state["players"][perspective]["hand"]
            if perspective is not None
            and 0 <= perspective < len(state["players"])
            and "hand" in state["players"][perspective]
            else []
        ),
        "turned_up_card": state["turned_up_card"],
        "current_trick": state["current_trick"],
        "team1_tricks": state["team1_tricks"],
        "team2_tricks": state["team2_tricks"],
        "going_alone": state["going_alone"],
    }

    return jsonify(formatted_state)


@game_bp.route("/games/<game_id>/move", methods=["POST"])
def play_move(game_id):
    """Play a card"""
    data = request.json
    card_str = data.get("card")

    if not card_str:
        return jsonify({"error": "Card required"}), 400

    try:
        card = Card.from_string(card_str)
    except Exception as e:
        return jsonify({"error": f"Invalid card: {str(e)}"}), 400

    # Load game from Redis
    game = load_game_from_redis(game_id)
    if not game:
        return jsonify({"error": "Game not found"}), 404

    try:
        # Play the card
        result = game.play_card(card)

        # Process AI turns after human move
        process_ai_turns(game)

        # Save updated game state
        save_game_to_redis(game)

        return jsonify({"success": True, "result": result, "state": game.get_state()})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@game_bp.route("/games/<game_id>/trump", methods=["POST"])
def call_trump(game_id):
    """Call trump suit"""
    data = request.json
    if not data:
        return jsonify({"error": "Request body required"}), 400

    suit_str = data.get("suit")
    go_alone = data.get("go_alone", False)
    pass_trump = data.get("pass", False)

    # Load game from Redis
    game = load_game_from_redis(game_id)
    if not game:
        return jsonify({"error": "Game not found"}), 404

    if pass_trump:
        # Player passes on trump
        try:
            game.pass_trump()
        except ValueError as e:
            return jsonify({"error": str(e)}), 400

        # Process AI turns after human pass
        process_ai_turns(game)

        save_game_to_redis(game)

        return jsonify({"success": True, "action": "pass", "state": game.get_state()})
    else:
        # Player calls trump
        suit = None
        if suit_str:
            try:
                suit = Suit.from_string(suit_str)
            except Exception as e:
                return jsonify({"error": f"Invalid suit: {str(e)}"}), 400

        try:
            game.call_trump(suit, go_alone)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400

        # Process AI turns after human calls trump
        process_ai_turns(game)

        save_game_to_redis(game)

        return jsonify(
            {
                "success": True,
                "action": "call_trump",
                "suit": suit_str,
                "go_alone": go_alone,
                "state": game.get_state(),
            }
        )


@game_bp.route("/games/<game_id>/discard", methods=["POST"])
def dealer_discard_card(game_id):
    """Dealer discards a card after picking up in round 1"""
    data = request.json
    card_str = data.get("card")

    if not card_str:
        return jsonify({"error": "Card required"}), 400

    try:
        card = Card.from_string(card_str)
    except Exception as e:
        return jsonify({"error": f"Invalid card: {str(e)}"}), 400

    # Load game from Redis
    game = load_game_from_redis(game_id)
    if not game:
        return jsonify({"error": "Game not found"}), 404

    try:
        # Discard the card
        game.dealer_discard(card)

        # Process AI turns after dealer discards
        process_ai_turns(game)

        # Save updated game state
        save_game_to_redis(game)

        return jsonify({"success": True, "state": game.get_state()})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@game_bp.route("/games/<game_id>/valid-moves", methods=["GET"])
def get_valid_moves(game_id):
    """Get valid moves for a player"""
    perspective = request.args.get("perspective", type=int)

    # Load game from Redis
    game = load_game_from_redis(game_id)
    if not game:
        return jsonify({"error": "Game not found"}), 404

    # Determine which player's valid moves to get
    player_position = (
        perspective if perspective is not None else game.state.current_player_position
    )

    # Get valid moves for the player
    valid_cards = game.get_valid_moves(player_position)

    # Get the player's hand for context
    player = game.state.get_player(player_position)
    hand = player.hand.copy()

    # Build response
    response = {
        "player_position": player_position,
        "valid_cards": [str(card) for card in valid_cards],
        "hand": [str(card) for card in hand],
        "must_follow_suit": len(valid_cards) < len(hand) if hand else False,
        "current_trick_lead_suit": (
            game.state.current_trick.lead_suit.value
            if game.state.current_trick and game.state.current_trick.lead_suit
            else None
        ),
        "trump_suit": game.state.trump.value if game.state.trump else None,
    }

    return jsonify(response)


@game_bp.route("/games/<game_id>/new-hand", methods=["POST"])
def start_new_hand(game_id):
    """Start a new hand after hand_complete phase"""
    # Load game from Redis
    game = load_game_from_redis(game_id)
    if not game:
        return jsonify({"error": "Game not found"}), 404

    # Check if we're in the right phase
    if game.state.phase == GamePhase.GAME_OVER:
        return jsonify({"error": "Game is over. Cannot start new hand."}), 400

    if game.state.phase != GamePhase.HAND_COMPLETE:
        return (
            jsonify(
                {"error": "Can only start new hand after current hand is complete"}
            ),
            400,
        )

    try:
        # Start a new hand
        game.start_new_hand()

        # Process AI turns for trump selection
        process_ai_turns(game)

        # Save updated game state
        save_game_to_redis(game)

        return jsonify({"success": True, "state": game.get_state()})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@game_bp.route("/games/<game_id>", methods=["DELETE"])
def delete_game(game_id):
    """Delete/abandon a game"""
    key = f"game:{game_id}"
    redis_client.delete(key)

    return jsonify({"success": True, "message": "Game deleted"})
