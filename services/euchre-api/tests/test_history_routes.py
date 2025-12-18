"""
Tests for history routes
"""

import pytest
import json


def test_list_games(client):
    """Test listing past games"""
    response = client.get('/api/history/')
    
    assert response.status_code == 200
    data = response.get_json()
    
    assert 'games' in data
    assert 'total' in data
    assert isinstance(data['games'], list)
    assert isinstance(data['total'], int)


def test_get_game_history(client):
    """Test getting complete history of a game"""
    game_id = 'test-game-123'
    
    response = client.get(f'/api/history/{game_id}')
    
    assert response.status_code == 200
    data = response.get_json()
    
    assert 'game_id' in data
    assert 'hands' in data
    assert 'moves' in data
    assert 'final_score' in data
    
    # Check final score structure
    assert 'team1' in data['final_score']
    assert 'team2' in data['final_score']


def test_get_game_history_structure(client):
    """Test game history has correct structure"""
    game_id = 'test-game-456'
    
    response = client.get(f'/api/history/{game_id}')
    
    assert response.status_code == 200
    data = response.get_json()
    
    assert data['game_id'] == game_id
    assert isinstance(data['hands'], list)
    assert isinstance(data['moves'], list)


def test_replay_game(client):
    """Test getting game replay data"""
    game_id = 'test-game-123'
    
    response = client.get(f'/api/history/{game_id}/replay')
    
    assert response.status_code == 200
    data = response.get_json()
    
    assert 'game_id' in data
    assert 'states' in data
    assert data['game_id'] == game_id
    assert isinstance(data['states'], list)


def test_replay_game_different_id(client):
    """Test replay with different game ID"""
    game_id = 'another-game-789'
    
    response = client.get(f'/api/history/{game_id}/replay')
    
    assert response.status_code == 200
    data = response.get_json()
    assert data['game_id'] == game_id
