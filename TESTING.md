# Testing Guide for EuchreBot

This document provides an overview of the testing infrastructure for the EuchreBot project.

## Euchre API Tests

The `euchre-api` service now includes a comprehensive pytest test suite.

### Test Coverage

The test suite covers:

- **Application Factory** (`test_app.py`)
  - App creation and configuration
  - Health endpoint
  - CORS configuration
  - Blueprint registration

- **Game Routes** (`test_game_routes.py`)
  - Game creation with various player types
  - Game state retrieval
  - Playing moves
  - Trump calling and passing
  - Game deletion

- **AI Routes** (`test_ai_routes.py`)
  - Listing available AI models
  - AI move predictions
  - AI automated play

- **History Routes** (`test_history_routes.py`)
  - Listing past games
  - Retrieving game history
  - Game replay functionality

### Running Tests Locally

1. **Install dependencies:**
   ```bash
   # Install euchre_core package
   cd shared/euchre_core
   pip install -e .
   
   # Install API dependencies
   cd ../../services/euchre-api
   pip install -r requirements.txt
   pip install -r requirements-test.txt
   ```

2. **Run tests:**
   ```bash
   cd services/euchre-api
   pytest
   ```

3. **Generate coverage report:**
   ```bash
   pytest --cov=app --cov-report=html
   open htmlcov/index.html  # View coverage in browser
   ```

### Continuous Integration

Tests are automatically run via GitHub Actions on:
- Pushes to `main` or `develop` branches
- Pull requests targeting `main` or `develop` branches

The CI workflow:
- Tests on Python 3.10 and 3.11
- Generates coverage reports
- Uploads coverage to Codecov
- Comments coverage metrics on pull requests

### Test Configuration

- **pytest.ini**: Pytest configuration with coverage settings
- **conftest.py**: Shared fixtures including Flask test client, mock Redis, and sample data
- **requirements-test.txt**: Test-specific dependencies

### Adding New Tests

When adding new routes or functionality:

1. Create test functions in the appropriate test file
2. Use existing fixtures from `conftest.py`
3. Follow the naming convention: `test_<functionality>`
4. Include docstrings describing what each test validates
5. Run tests locally before committing

### Test Fixtures Available

- `app` - Flask application configured for testing
- `client` - Flask test client for HTTP requests
- `mock_redis` - Mocked Redis client
- `sample_game_data` - Sample game creation payload
- `sample_card` - Sample card string
- `sample_suit` - Sample suit string

## Future Testing Enhancements

Potential areas for expansion:

- Integration tests with real database
- End-to-end tests for complete game flows
- Performance/load testing
- Tests for other services (web-ui, ai-trainer, cli)
- Contract testing between services
