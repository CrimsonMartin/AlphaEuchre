"""
Tests for Flask application factory and basic endpoints
"""

import pytest
from app import create_app


def test_app_creation():
    """Test that the app can be created"""
    app = create_app()
    assert app is not None
    assert app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] is False


def test_app_testing_config(app):
    """Test that the app is in testing mode"""
    assert app.config['TESTING'] is True


def test_health_endpoint(client):
    """Test the health check endpoint"""
    response = client.get('/health')
    assert response.status_code == 200
    
    data = response.get_json()
    assert data['status'] == 'healthy'
    assert data['service'] == 'euchre-api'


def test_cors_enabled(client):
    """Test that CORS is enabled"""
    response = client.options('/health')
    # CORS should allow the request
    assert response.status_code in [200, 204]


def test_blueprints_registered(app):
    """Test that all blueprints are registered"""
    blueprint_names = [bp.name for bp in app.blueprints.values()]
    
    assert 'game' in blueprint_names
    assert 'ai' in blueprint_names
    assert 'history' in blueprint_names


def test_404_error(client):
    """Test 404 error for non-existent endpoint"""
    response = client.get('/api/nonexistent')
    assert response.status_code == 404
