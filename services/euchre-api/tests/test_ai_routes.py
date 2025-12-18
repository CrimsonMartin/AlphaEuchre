"""
Tests for AI routes
"""

import pytest
import json


def test_list_models(client):
    """Test listing available AI models"""
    response = client.get('/api/ai/models')
    
    assert response.status_code == 200
    data = response.get_json()
    
    assert 'models' in data
    assert isinstance(data['models'], list)
    assert len(data['models']) > 0
    
    # Check first model structure
    model = data['models'][0]
    assert 'id' in model
    assert 'name' in model
    assert 'type' in model
    assert 'active' in model


def test_predict_move_default_model(client):
    """Test AI move prediction with default model"""
    prediction_data = {
        'game_state': {
            'current_player': 0,
            'hand': ['9H', '10H', 'JH']
        }
    }
    
    response = client.post(
        '/api/ai/predict',
        data=json.dumps(prediction_data),
        content_type='application/json'
    )
    
    assert response.status_code == 200
    data = response.get_json()
    
    assert 'recommended_card' in data
    assert 'confidence' in data
    assert 'model_id' in data
    assert data['model_id'] == 'random'


def test_predict_move_specific_model(client):
    """Test AI move prediction with specific model"""
    prediction_data = {
        'game_state': {
            'current_player': 0,
            'hand': ['9H', '10H', 'JH']
        },
        'model_id': 'neural_net'
    }
    
    response = client.post(
        '/api/ai/predict',
        data=json.dumps(prediction_data),
        content_type='application/json'
    )
    
    assert response.status_code == 200
    data = response.get_json()
    assert data['model_id'] == 'neural_net'


def test_ai_play_move(client):
    """Test having AI play a move"""
    play_data = {
        'game_id': 'test-game-123'
    }
    
    response = client.post(
        '/api/ai/models/random/play',
        data=json.dumps(play_data),
        content_type='application/json'
    )
    
    assert response.status_code == 200
    data = response.get_json()
    
    assert data['success'] is True
    assert 'model_id' in data
    assert 'card_played' in data
    assert data['model_id'] == 'random'


def test_ai_play_move_different_model(client):
    """Test AI play with different model"""
    play_data = {
        'game_id': 'test-game-456'
    }
    
    response = client.post(
        '/api/ai/models/neural_net/play',
        data=json.dumps(play_data),
        content_type='application/json'
    )
    
    assert response.status_code == 200
    data = response.get_json()
    assert data['model_id'] == 'neural_net'
