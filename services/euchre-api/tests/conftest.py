"""
Pytest configuration and fixtures for euchre-api tests
"""

import pytest
from unittest.mock import MagicMock
from app import create_app, db


@pytest.fixture
def app():
    """Create and configure a test Flask application"""
    test_config = {
        'TESTING': True,
        'SQLALCHEMY_DATABASE_URI': 'sqlite:///:memory:',
        'SQLALCHEMY_TRACK_MODIFICATIONS': False,
        'SECRET_KEY': 'test-secret-key'
    }
    
    app = create_app(test_config)
    
    with app.app_context():
        db.create_all()
        yield app
        db.session.remove()
        db.drop_all()


@pytest.fixture
def client(app):
    """Create a test client for the Flask application"""
    return app.test_client()


@pytest.fixture
def runner(app):
    """Create a test CLI runner"""
    return app.test_cli_runner()


@pytest.fixture
def mock_redis(monkeypatch):
    """Mock Redis client"""
    mock_redis_client = MagicMock()
    
    # Mock Redis methods
    mock_redis_client.get.return_value = None
    mock_redis_client.set.return_value = True
    mock_redis_client.delete.return_value = 1
    
    # Patch the redis_client in the app module
    import app as app_module
    monkeypatch.setattr(app_module, 'redis_client', mock_redis_client)
    
    # Also patch it in game_routes
    from app.routes import game_routes
    monkeypatch.setattr(game_routes, 'redis_client', mock_redis_client)
    
    return mock_redis_client


@pytest.fixture
def sample_game_data():
    """Sample game creation data"""
    return {
        'players': [
            {'name': 'Player 1', 'type': 'human'},
            {'name': 'Player 2', 'type': 'random_ai'},
            {'name': 'Player 3', 'type': 'human'},
            {'name': 'Player 4', 'type': 'random_ai'}
        ]
    }


@pytest.fixture
def sample_card():
    """Sample card string"""
    return '9H'


@pytest.fixture
def sample_suit():
    """Sample suit string"""
    return 'hearts'
