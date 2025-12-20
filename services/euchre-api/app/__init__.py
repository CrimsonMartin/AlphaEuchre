"""
Euchre API Flask Application
"""

import os
from flask import Flask
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
import redis

# Initialize extensions
db = SQLAlchemy()
redis_client = None


def create_app(test_config=None):
    """Application factory pattern"""
    app = Flask(__name__)
    
    # Configuration
    if test_config is None:
        # Production/development configuration
        app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-key')
        app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv(
            'DATABASE_URL', 
            'postgresql://euchre:euchre_dev_pass@postgres:5432/euchrebot'
        )
        app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    else:
        # Test configuration
        app.config.update(test_config)
    
    # Initialize extensions
    db.init_app(app)
    CORS(app)
    
    # Initialize Redis
    global redis_client
    if test_config and 'TESTING' in test_config and test_config['TESTING']:
        # Use fakeredis for testing
        import fakeredis
        redis_client = fakeredis.FakeStrictRedis()
    else:
        redis_url = os.getenv('REDIS_URL', 'redis://redis:6379/0')
        redis_client = redis.from_url(redis_url)
    
    # Register blueprints
    from app.routes.game_routes import game_bp
    from app.routes.ai_routes import ai_bp
    from app.routes.history_routes import history_bp
    
    app.register_blueprint(game_bp, url_prefix='/api')
    app.register_blueprint(ai_bp, url_prefix='/api/ai')
    app.register_blueprint(history_bp, url_prefix='/api/history')
    
    # Health check endpoint
    @app.route('/health')
    def health():
        return {'status': 'healthy', 'service': 'euchre-api'}
    
    return app
