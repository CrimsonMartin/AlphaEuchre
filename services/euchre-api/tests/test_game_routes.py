"""
Tests for game routes
"""

import pytest
import json
from unittest.mock import patch, MagicMock


def test_create_game_success(client, mock_redis, sample_game_data):
    """Test successful game creation"""
    response = client.post(
        '/api/games',
        data=json.dumps(sample_game_data),
        content_type='application/json'
    )
    
    assert response.status_code == 201
    data = response.get_json()
    
    assert 'game_id' in data
    assert 'state' in data
    assert data['game_id'] is not None
    
    # Verify Redis was called to save the game
    mock_redis.set.assert_called_once()


def test_create_game_invalid_player_count(client, mock_redis):
    """Test game creation with invalid number of players"""
    invalid_data = {
        'players': [
            {'name': 'Player 1', 'type': 'human'},
            {'name': 'Player 2', 'type': 'human'}
        ]
    }
    
    response = client.post(
        '/api/games',
        data=json.dumps(invalid_data),
        content_type='application/json'
    )
    
    assert response.status_code == 400
    data = response.get_json()
    assert 'error' in data
    assert 'Exactly 4 players required' in data['error']


def test_create_game_with_different_player_types(client, mock_redis):
    """Test game creation with different player types"""
    game_data = {
        'players': [
            {'name': 'Human', 'type': 'human'},
            {'name': 'Random AI', 'type': 'random_ai'},
            {'name': 'Neural AI', 'type': 'neural_net_ai'},
            {'name': 'Another Human', 'type': 'human'}
        ]
    }
    
    response = client.post(
        '/api/games',
        data=json.dumps(game_data),
        content_type='application/json'
    )
    
    assert response.status_code == 201


def test_get_game(client, mock_redis):
    """Test getting game state"""
    game_id = 'test-game-123'
    
    response = client.get(f'/api/games/{game_id}')
    
    assert response.status_code == 200
    data = response.get_json()
    assert 'game_id' in data


def test_get_game_with_perspective(client, mock_redis):
    """Test getting game state with player perspective"""
    game_id = 'test-game-123'
    
    response = client.get(f'/api/games/{game_id}?perspective=0')
    
    assert response.status_code == 200


def test_play_move_success(client, mock_redis, sample_card):
    """Test playing a card"""
    game_id = 'test-game-123'
    move_data = {'card': sample_card}
    
    response = client.post(
        f'/api/games/{game_id}/move',
        data=json.dumps(move_data),
        content_type='application/json'
    )
    
    assert response.status_code == 200
    data = response.get_json()
    assert data['success'] is True
    assert data['card'] == sample_card


def test_play_move_missing_card(client, mock_redis):
    """Test playing a move without specifying a card"""
    game_id = 'test-game-123'
    
    response = client.post(
        f'/api/games/{game_id}/move',
        data=json.dumps({}),
        content_type='application/json'
    )
    
    assert response.status_code == 400
    data = response.get_json()
    assert 'error' in data
    assert 'Card required' in data['error']


def test_call_trump_with_suit(client, mock_redis, sample_suit):
    """Test calling trump with a suit"""
    game_id = 'test-game-123'
    trump_data = {'suit': sample_suit, 'go_alone': False}
    
    response = client.post(
        f'/api/games/{game_id}/trump',
        data=json.dumps(trump_data),
        content_type='application/json'
    )
    
    assert response.status_code == 200
    data = response.get_json()
    assert data['success'] is True
    assert data['action'] == 'call_trump'
    assert data['suit'] == sample_suit
    assert data['go_alone'] is False


def test_call_trump_go_alone(client, mock_redis, sample_suit):
    """Test calling trump and going alone"""
    game_id = 'test-game-123'
    trump_data = {'suit': sample_suit, 'go_alone': True}
    
    response = client.post(
        f'/api/games/{game_id}/trump',
        data=json.dumps(trump_data),
        content_type='application/json'
    )
    
    assert response.status_code == 200
    data = response.get_json()
    assert data['go_alone'] is True


def test_pass_trump(client, mock_redis):
    """Test passing on trump"""
    game_id = 'test-game-123'
    trump_data = {'pass': True}
    
    response = client.post(
        f'/api/games/{game_id}/trump',
        data=json.dumps(trump_data),
        content_type='application/json'
    )
    
    assert response.status_code == 200
    data = response.get_json()
    assert data['success'] is True
    assert data['action'] == 'pass'


def test_delete_game(client, mock_redis):
    """Test deleting a game"""
    game_id = 'test-game-123'
    
    response = client.delete(f'/api/games/{game_id}')
    
    assert response.status_code == 200
    data = response.get_json()
    assert data['success'] is True
    
    # Verify Redis delete was called
    mock_redis.delete.assert_called_once()
