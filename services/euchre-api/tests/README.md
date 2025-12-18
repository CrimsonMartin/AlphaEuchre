# Euchre API Tests

This directory contains the pytest test suite for the euchre-api service.

## Test Structure

- `conftest.py` - Pytest fixtures and configuration
- `test_app.py` - Tests for Flask application factory and basic endpoints
- `test_game_routes.py` - Tests for game management routes
- `test_ai_routes.py` - Tests for AI model routes
- `test_history_routes.py` - Tests for game history routes

## Running Tests

### Prerequisites

Install test dependencies:

```bash
cd services/euchre-api
pip install -r requirements.txt
pip install -r requirements-test.txt
```

Also install the shared euchre_core package:

```bash
cd ../../shared/euchre_core
pip install -e .
```

### Run All Tests

```bash
cd services/euchre-api
pytest
```

### Run Specific Test File

```bash
pytest tests/test_game_routes.py
```

### Run Specific Test

```bash
pytest tests/test_game_routes.py::test_create_game_success
```

### Run with Coverage Report

```bash
pytest --cov=app --cov-report=html
```

Then open `htmlcov/index.html` in your browser to view the coverage report.

### Run Tests in Verbose Mode

```bash
pytest -v
```

### Run Tests with Output

```bash
pytest -s
```

## Test Fixtures

The following fixtures are available in `conftest.py`:

- `app` - Flask application instance configured for testing
- `client` - Flask test client for making HTTP requests
- `runner` - Flask CLI test runner
- `mock_redis` - Mocked Redis client
- `sample_game_data` - Sample game creation data
- `sample_card` - Sample card string
- `sample_suit` - Sample suit string

## CI/CD

Tests are automatically run on GitHub Actions when:
- Code is pushed to `main` or `develop` branches
- Pull requests are created targeting `main` or `develop` branches

The workflow runs tests on Python 3.10 and 3.11, and generates coverage reports.
