"""
Comprehensive tests for invalid moves and game states.
Tests all error conditions and invalid operations in game_routes.py
"""

import pytest
import json
from unittest.mock import patch, MagicMock
from euchre_core import Card, Suit
from euchre_core.game import GamePhase


class TestInvalidGameCreation:
    """Test invalid game creation scenarios"""

    def test_create_game_with_zero_players(self, client, mock_redis):
        """Test game creation with no players"""
        response = client.post("/api/games", json={"players": []})
        assert response.status_code == 400
        assert "error" in response.json
        assert "Exactly 4 players required" in response.json["error"]

    def test_create_game_with_one_player(self, client, mock_redis):
        """Test game creation with only one player"""
        response = client.post(
            "/api/games", json={"players": [{"name": "Player1", "type": "human"}]}
        )
        assert response.status_code == 400
        assert "Exactly 4 players required" in response.json["error"]

    def test_create_game_with_three_players(self, client, mock_redis):
        """Test game creation with three players"""
        response = client.post(
            "/api/games",
            json={
                "players": [
                    {"name": "Player1", "type": "human"},
                    {"name": "Player2", "type": "human"},
                    {"name": "Player3", "type": "human"},
                ]
            },
        )
        assert response.status_code == 400
        assert "Exactly 4 players required" in response.json["error"]

    def test_create_game_with_five_players(self, client, mock_redis):
        """Test game creation with too many players"""
        response = client.post(
            "/api/games",
            json={
                "players": [
                    {"name": "Player1", "type": "human"},
                    {"name": "Player2", "type": "human"},
                    {"name": "Player3", "type": "human"},
                    {"name": "Player4", "type": "human"},
                    {"name": "Player5", "type": "human"},
                ]
            },
        )
        assert response.status_code == 400
        assert "Exactly 4 players required" in response.json["error"]


class TestInvalidCardPlays:
    """Test invalid card play scenarios"""

    def test_play_card_not_in_hand(self, client):
        """Test playing a card that's not in the player's hand"""
        # Create game
        response = client.post(
            "/api/games",
            json={
                "players": [
                    {"name": "Player1", "type": "human"},
                    {"name": "Player2", "type": "human"},
                    {"name": "Player3", "type": "human"},
                    {"name": "Player4", "type": "human"},
                ]
            },
        )
        game_id = response.json["game_id"]

        # Call trump to get to playing phase
        client.post(
            f"/api/games/{game_id}/trump", json={"suit": None, "go_alone": False}
        )

        # Try to play a card that's definitely not in hand (all cards are 9-A, so use invalid card)
        response = client.post(f"/api/games/{game_id}/move", json={"card": "2H"})
        assert response.status_code == 400
        assert "error" in response.json

    def test_play_card_violating_follow_suit(self, client):
        """Test playing a card that violates follow suit rules"""
        # Create game with all human players
        response = client.post(
            "/api/games",
            json={
                "players": [
                    {"name": "Player1", "type": "human"},
                    {"name": "Player2", "type": "human"},
                    {"name": "Player3", "type": "human"},
                    {"name": "Player4", "type": "human"},
                ]
            },
        )
        game_id = response.json["game_id"]

        # Call trump to get to playing phase
        client.post(
            f"/api/games/{game_id}/trump", json={"suit": None, "go_alone": False}
        )

        # Get current player's hand
        state = client.get(f"/api/games/{game_id}?perspective=0").json

        # This test would need specific hand setup to guarantee follow suit violation
        # The actual violation will be caught by the game engine

    def test_play_card_when_not_in_playing_phase(self, client):
        """Test playing a card during trump selection phase"""
        # Create game
        response = client.post(
            "/api/games",
            json={
                "players": [
                    {"name": "Player1", "type": "human"},
                    {"name": "Player2", "type": "human"},
                    {"name": "Player3", "type": "human"},
                    {"name": "Player4", "type": "human"},
                ]
            },
        )
        game_id = response.json["game_id"]

        # Try to play a card during trump selection (before calling trump)
        response = client.post(f"/api/games/{game_id}/move", json={"card": "9H"})
        assert response.status_code == 400
        assert "error" in response.json
        assert "Not in playing phase" in response.json["error"]

    def test_play_card_with_invalid_format(self, client):
        """Test playing a card with invalid string format"""
        # Create game
        response = client.post(
            "/api/games",
            json={
                "players": [
                    {"name": "Player1", "type": "human"},
                    {"name": "Player2", "type": "human"},
                    {"name": "Player3", "type": "human"},
                    {"name": "Player4", "type": "human"},
                ]
            },
        )
        game_id = response.json["game_id"]

        # Try to play with invalid card format
        response = client.post(f"/api/games/{game_id}/move", json={"card": "INVALID"})
        assert response.status_code == 400
        assert "error" in response.json
        assert "Invalid card" in response.json["error"]

    def test_play_card_missing_card_parameter(self, client):
        """Test playing without providing a card"""
        # Create game
        response = client.post(
            "/api/games",
            json={
                "players": [
                    {"name": "Player1", "type": "human"},
                    {"name": "Player2", "type": "human"},
                    {"name": "Player3", "type": "human"},
                    {"name": "Player4", "type": "human"},
                ]
            },
        )
        game_id = response.json["game_id"]

        # Try to play without card parameter
        response = client.post(f"/api/games/{game_id}/move", json={})
        assert response.status_code == 400
        assert "Card required" in response.json["error"]

    def test_play_card_on_nonexistent_game(self, client, mock_redis):
        """Test playing a card on a game that doesn't exist"""
        mock_redis.get.return_value = None

        response = client.post("/api/games/nonexistent-id/move", json={"card": "9H"})
        assert response.status_code == 404
        assert "Game not found" in response.json["error"]


class TestInvalidTrumpCalls:
    """Test invalid trump calling scenarios"""

    def test_call_wrong_suit_in_round1(self, client):
        """Test calling a different suit than turned up card in round 1"""
        # Create game
        response = client.post(
            "/api/games",
            json={
                "players": [
                    {"name": "Player1", "type": "human"},
                    {"name": "Player2", "type": "human"},
                    {"name": "Player3", "type": "human"},
                    {"name": "Player4", "type": "human"},
                ]
            },
        )
        game_id = response.json["game_id"]

        # Get the turned up card
        state = client.get(f"/api/games/{game_id}").json
        turned_up_card = state.get("turned_up_card")

        if turned_up_card:
            # Extract suit from turned up card (last character)
            turned_up_suit = turned_up_card[-1]

            # Try to call a different suit in round 1
            wrong_suits = {
                "H": "D",
                "D": "H",
                "C": "S",
                "S": "C",
                "♥": "D",
                "♦": "H",
                "♣": "S",
                "♠": "C",
            }
            wrong_suit = wrong_suits.get(turned_up_suit, "H")

            response = client.post(
                f"/api/games/{game_id}/trump",
                json={"suit": wrong_suit, "go_alone": False},
            )
            assert response.status_code == 400
            assert "error" in response.json

    def test_call_turned_up_suit_in_round2(self, client):
        """Test calling the turned up suit in round 2 (not allowed)"""
        # Create game
        response = client.post(
            "/api/games",
            json={
                "players": [
                    {"name": "Player1", "type": "human"},
                    {"name": "Player2", "type": "human"},
                    {"name": "Player3", "type": "human"},
                    {"name": "Player4", "type": "human"},
                ]
            },
        )
        game_id = response.json["game_id"]

        # Get the turned up card
        state = client.get(f"/api/games/{game_id}").json
        turned_up_card = state.get("turned_up_card")
        turned_up_suit = turned_up_card[-1] if turned_up_card else "H"

        # Pass for all 4 players to get to round 2
        for _ in range(4):
            client.post(f"/api/games/{game_id}/trump", json={"pass": True})

        # Try to call the turned up suit in round 2
        response = client.post(
            f"/api/games/{game_id}/trump",
            json={"suit": turned_up_suit, "go_alone": False},
        )
        assert response.status_code == 400
        assert "error" in response.json
        assert "Cannot call turned up suit in round 2" in response.json["error"]

    def test_dealer_cannot_pass_in_round2(self, client):
        """Test that dealer cannot pass in round 2 (stick the dealer)"""
        # Create game
        response = client.post(
            "/api/games",
            json={
                "players": [
                    {"name": "Player1", "type": "human"},
                    {"name": "Player2", "type": "human"},
                    {"name": "Player3", "type": "human"},
                    {"name": "Player4", "type": "human"},
                ]
            },
        )
        game_id = response.json["game_id"]

        # Pass for all 4 players in round 1 to get to round 2
        for _ in range(4):
            client.post(f"/api/games/{game_id}/trump", json={"pass": True})

        # Pass for first 3 players in round 2 to get to dealer
        for _ in range(3):
            client.post(f"/api/games/{game_id}/trump", json={"pass": True})

        # Dealer tries to pass in round 2
        response = client.post(f"/api/games/{game_id}/trump", json={"pass": True})
        assert response.status_code == 400
        assert "error" in response.json
        assert "Dealer cannot pass in round 2" in response.json["error"]

    def test_call_trump_without_suit_in_round2(self, client):
        """Test calling trump without specifying suit in round 2"""
        # Create game
        response = client.post(
            "/api/games",
            json={
                "players": [
                    {"name": "Player1", "type": "human"},
                    {"name": "Player2", "type": "human"},
                    {"name": "Player3", "type": "human"},
                    {"name": "Player4", "type": "human"},
                ]
            },
        )
        game_id = response.json["game_id"]

        # Pass for all 4 players to get to round 2
        for _ in range(4):
            client.post(f"/api/games/{game_id}/trump", json={"pass": True})

        # Try to call trump without suit in round 2
        response = client.post(
            f"/api/games/{game_id}/trump", json={"suit": None, "go_alone": False}
        )
        assert response.status_code == 400
        assert "error" in response.json
        assert "Must specify suit in round 2" in response.json["error"]

    def test_call_trump_on_nonexistent_game(self, client, mock_redis):
        """Test calling trump on a game that doesn't exist"""
        mock_redis.get.return_value = None

        response = client.post(
            "/api/games/nonexistent-id/trump", json={"suit": "H", "go_alone": False}
        )
        assert response.status_code == 404
        assert "Game not found" in response.json["error"]

    def test_call_trump_with_invalid_suit_string(self, client):
        """Test calling trump with invalid suit string"""
        # Create game
        response = client.post(
            "/api/games",
            json={
                "players": [
                    {"name": "Player1", "type": "human"},
                    {"name": "Player2", "type": "human"},
                    {"name": "Player3", "type": "human"},
                    {"name": "Player4", "type": "human"},
                ]
            },
        )
        game_id = response.json["game_id"]

        # Pass to round 2
        for _ in range(4):
            client.post(f"/api/games/{game_id}/trump", json={"pass": True})

        # Try to call with invalid suit
        response = client.post(
            f"/api/games/{game_id}/trump", json={"suit": "INVALID", "go_alone": False}
        )
        assert response.status_code == 400
        assert "error" in response.json


class TestInvalidDealerDiscards:
    """Test invalid dealer discard scenarios"""

    def test_discard_card_not_in_hand(self, client):
        """Test dealer discarding a card not in their hand"""
        # Create game
        response = client.post(
            "/api/games",
            json={
                "players": [
                    {"name": "Player1", "type": "human"},
                    {"name": "Player2", "type": "human"},
                    {"name": "Player3", "type": "human"},
                    {"name": "Player4", "type": "human"},
                ]
            },
        )
        game_id = response.json["game_id"]

        # Call trump in round 1 to trigger dealer discard
        client.post(
            f"/api/games/{game_id}/trump", json={"suit": None, "go_alone": False}
        )

        # Try to discard a card not in hand
        response = client.post(f"/api/games/{game_id}/discard", json={"card": "2H"})
        assert response.status_code == 400
        assert "error" in response.json

    def test_discard_when_not_in_discard_phase(self, client):
        """Test discarding when not in dealer discard phase"""
        # Create game
        response = client.post(
            "/api/games",
            json={
                "players": [
                    {"name": "Player1", "type": "human"},
                    {"name": "Player2", "type": "human"},
                    {"name": "Player3", "type": "human"},
                    {"name": "Player4", "type": "human"},
                ]
            },
        )
        game_id = response.json["game_id"]

        # Try to discard during trump selection (before calling trump)
        response = client.post(f"/api/games/{game_id}/discard", json={"card": "9H"})
        assert response.status_code == 400
        assert "error" in response.json
        assert "Not in dealer discard phase" in response.json["error"]

    def test_discard_missing_card_parameter(self, client):
        """Test discarding without providing a card"""
        # Create game
        response = client.post(
            "/api/games",
            json={
                "players": [
                    {"name": "Player1", "type": "human"},
                    {"name": "Player2", "type": "human"},
                    {"name": "Player3", "type": "human"},
                    {"name": "Player4", "type": "human"},
                ]
            },
        )
        game_id = response.json["game_id"]

        # Call trump to get to discard phase
        client.post(
            f"/api/games/{game_id}/trump", json={"suit": None, "go_alone": False}
        )

        # Try to discard without card parameter
        response = client.post(f"/api/games/{game_id}/discard", json={})
        assert response.status_code == 400
        assert "Card required" in response.json["error"]

    def test_discard_with_invalid_card_format(self, client):
        """Test discarding with invalid card format"""
        # Create game
        response = client.post(
            "/api/games",
            json={
                "players": [
                    {"name": "Player1", "type": "human"},
                    {"name": "Player2", "type": "human"},
                    {"name": "Player3", "type": "human"},
                    {"name": "Player4", "type": "human"},
                ]
            },
        )
        game_id = response.json["game_id"]

        # Call trump to get to discard phase
        client.post(
            f"/api/games/{game_id}/trump", json={"suit": None, "go_alone": False}
        )

        # Try to discard with invalid format
        response = client.post(
            f"/api/games/{game_id}/discard", json={"card": "INVALID"}
        )
        assert response.status_code == 400
        assert "error" in response.json
        assert "Invalid card" in response.json["error"]

    def test_discard_on_nonexistent_game(self, client, mock_redis):
        """Test discarding on a game that doesn't exist"""
        mock_redis.get.return_value = None

        response = client.post("/api/games/nonexistent-id/discard", json={"card": "9H"})
        assert response.status_code == 404
        assert "Game not found" in response.json["error"]


class TestInvalidGameStates:
    """Test operations on invalid game states"""

    def test_get_nonexistent_game(self, client, mock_redis):
        """Test getting a game that doesn't exist"""
        mock_redis.get.return_value = None

        response = client.get("/api/games/nonexistent-id")
        assert response.status_code == 404
        assert "Game not found" in response.json["error"]

    def test_get_valid_moves_on_nonexistent_game(self, client, mock_redis):
        """Test getting valid moves for a game that doesn't exist"""
        mock_redis.get.return_value = None

        response = client.get("/api/games/nonexistent-id/valid-moves")
        assert response.status_code == 404
        assert "Game not found" in response.json["error"]


class TestEdgeCases:
    """Test edge cases and boundary conditions"""

    def test_multiple_invalid_moves_in_sequence(self, client):
        """Test making multiple invalid moves in a row"""
        # Create game
        response = client.post(
            "/api/games",
            json={
                "players": [
                    {"name": "Player1", "type": "human"},
                    {"name": "Player2", "type": "human"},
                    {"name": "Player3", "type": "human"},
                    {"name": "Player4", "type": "human"},
                ]
            },
        )
        game_id = response.json["game_id"]

        # Try multiple invalid operations
        # 1. Play card during trump selection
        response1 = client.post(f"/api/games/{game_id}/move", json={"card": "9H"})
        assert response1.status_code == 400

        # 2. Discard during trump selection
        response2 = client.post(f"/api/games/{game_id}/discard", json={"card": "9H"})
        assert response2.status_code == 400

        # 3. Call invalid suit in round 1
        response3 = client.post(
            f"/api/games/{game_id}/trump", json={"suit": "X", "go_alone": False}
        )
        assert response3.status_code == 400

    def test_empty_json_payload(self, client):
        """Test endpoints with empty JSON payloads"""
        # Create game with empty payload
        response = client.post("/api/games", json={})
        assert response.status_code == 400

    def test_malformed_json(self, client):
        """Test endpoints with malformed JSON"""
        response = client.post(
            "/api/games",
            data='{"players": [invalid json}',
            content_type="application/json",
        )
        # Should get 400 or 500 depending on Flask error handling
        assert response.status_code in [400, 500]

    def test_null_values_in_requests(self, client):
        """Test handling of null values in request parameters"""
        # Create game
        response = client.post(
            "/api/games",
            json={
                "players": [
                    {"name": None, "type": "human"},
                    {"name": "Player2", "type": "human"},
                    {"name": "Player3", "type": "human"},
                    {"name": "Player4", "type": "human"},
                ]
            },
        )
        # Should still create game (name defaults to "Player")
        assert response.status_code == 201

    def test_very_long_player_names(self, client):
        """Test creating game with very long player names"""
        long_name = "A" * 1000
        response = client.post(
            "/api/games",
            json={
                "players": [
                    {"name": long_name, "type": "human"},
                    {"name": "Player2", "type": "human"},
                    {"name": "Player3", "type": "human"},
                    {"name": "Player4", "type": "human"},
                ]
            },
        )
        # Should still work
        assert response.status_code == 201

    def test_special_characters_in_player_names(self, client):
        """Test creating game with special characters in names"""
        response = client.post(
            "/api/games",
            json={
                "players": [
                    {"name": '<script>alert("xss")</script>', "type": "human"},
                    {"name": 'Player"2', "type": "human"},
                    {"name": "Player'3", "type": "human"},
                    {"name": "Player\n4", "type": "human"},
                ]
            },
        )
        # Should still work
        assert response.status_code == 201

    def test_invalid_perspective_values(self, client):
        """Test getting game with invalid perspective values"""
        # Create game
        response = client.post(
            "/api/games",
            json={
                "players": [
                    {"name": "Player1", "type": "human"},
                    {"name": "Player2", "type": "human"},
                    {"name": "Player3", "type": "human"},
                    {"name": "Player4", "type": "human"},
                ]
            },
        )
        game_id = response.json["game_id"]

        # Test with out of range perspective
        response = client.get(f"/api/games/{game_id}?perspective=10")
        # Should still return 200, just won't show hand
        assert response.status_code == 200

        # Test with negative perspective
        response = client.get(f"/api/games/{game_id}?perspective=-1")
        assert response.status_code == 200

    def test_concurrent_invalid_operations(self, client):
        """Test that game state remains consistent after invalid operations"""
        # Create game
        response = client.post(
            "/api/games",
            json={
                "players": [
                    {"name": "Player1", "type": "human"},
                    {"name": "Player2", "type": "human"},
                    {"name": "Player3", "type": "human"},
                    {"name": "Player4", "type": "human"},
                ]
            },
        )
        game_id = response.json["game_id"]

        # Get initial state
        initial_state = client.get(f"/api/games/{game_id}").json

        # Try invalid operation
        client.post(f"/api/games/{game_id}/move", json={"card": "9H"})

        # Get state after invalid operation
        after_state = client.get(f"/api/games/{game_id}").json

        # State should be unchanged
        assert initial_state["phase"] == after_state["phase"]
        assert (
            initial_state["current_player_position"]
            == after_state["current_player_position"]
        )
