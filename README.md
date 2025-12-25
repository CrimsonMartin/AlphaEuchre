# AlphaEuchre

Euchre strategy by exclusively reinforcement learning, because why not.

Everything below written by AI, so read at your own risk:

## 🎯 Features

- **Complete Euchre Game Engine** - Full rule implementation including trump selection, going alone, and proper trick-taking
- **Multiple Interfaces**:
  - CLI - Terminal-based interactive gameplay
  - Web UI - Browser-based game visualization
  - REST API - For programmatic access
- **AI Training Framework**:
  - PyTorch neural networks for card playing strategy
  - Genetic algorithm for evolving better players
  - CUDA GPU support for accelerated training
- **Database Storage** - PostgreSQL database for game history and replay
- **Containerized Architecture** - 8 separate Docker containers for scalability

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     EuchreBot System                         │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │   CLI    │  │  Web UI  │  │ Euchre   │  │    AI    │   │
│  │ Terminal │  │  Flask   │  │   API    │  │ Trainer  │   │
│  │          │  │ :5001    │  │  Flask   │  │ PyTorch  │   │
│  │          │  │          │  │  :5000   │  │  :5002   │   │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘   │
│       │             │              │              │          │
│       └─────────────┴──────────────┴──────────────┘          │
│                         │                                    │
│              ┌──────────┴──────────┐                        │
│              │                     │                        │
│         ┌────▼────┐          ┌────▼────┐                   │
│         │ Redis   │          │Postgres │                   │
│         │ Cache   │          │Database │                   │
│         │ :6379   │          │  :5432  │                   │
│         └─────────┘          └─────────┘                   │
│                                                              │
│         ┌─────────────────────────────┐                    │
│         │   Shared: euchre_core       │                    │
│         │   Core Game Engine          │                    │
│         └─────────────────────────────┘                    │
└──────────────────────────────────────────────────────────────┘
```

## 🐳 Docker Services

| Service | Port | Purpose |
|---------|------|---------|
| **postgres** | 5432 | PostgreSQL database for game history |
| **redis** | 6379 | Session storage and caching |
| **euchre-api** | 5000 | Flask REST API (game engine) |
| **web-ui** | 5001 | Flask web interface |
| **cli** | - | Terminal-based game interface |
| **ai-trainer** | 5002 | PyTorch neural network training |
| **pgadmin** | 5050 | Database admin interface (optional) |

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose
- NVIDIA Docker runtime (for GPU training - optional)
- Git
- VS Code with Dev Containers extension (recommended)

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/YourUsername/EuchreBot.git
   cd EuchreBot
   ```

2. **Start all services**
   ```bash
   docker-compose up -d
   ```

3. **Access the interfaces**
   - Web UI: http://localhost:5001
   - API: http://localhost:5000
   - PgAdmin: http://localhost:5050
   - CLI: `docker-compose run cli`

### Using the Dev Container

The project includes a full development container with PyTorch, CUDA support, and Docker-outside-of-Docker:

1. Open the project in VS Code
2. Click "Reopen in Container" when prompted
3. The devcontainer will build with all dependencies
4. Your host SSH keys are mounted for git authentication

## 🎮 Usage

### Playing via CLI

```bash
# Run the CLI container
docker-compose run cli

# Or using docker directly
docker-compose up cli
```

The CLI provides a colorful, interactive terminal interface for playing Euchre.

### Playing via Web UI

1. Navigate to http://localhost:5001
2. Click "New Game"
3. Configure players (Human vs AI)
4. Play!

### Using the API

#### Create a Game
```bash
curl -X POST http://localhost:5000/api/games \
  -H "Content-Type: application/json" \
  -d '{
    "players": [
      {"name": "Player 1", "type": "human"},
      {"name": "AI 1", "type": "random_ai"},
      {"name": "AI 2", "type": "random_ai"},
      {"name": "AI 3", "type": "random_ai"}
    ]
  }'
```

#### Get Game State
```bash
curl http://localhost:5000/api/games/{game_id}
```

#### Play a Card
```bash
curl -X POST http://localhost:5000/api/games/{game_id}/move \
  -H "Content-Type: application/json" \
  -d '{"card": "9H"}'
```

## 🤖 AI Training

### Start a Training Run

```bash
curl -X POST http://localhost:5002/api/train/start \
  -H "Content-Type: application/json" \
  -d '{
    "population_size": 20,
    "generations": 10
  }'
```

### Check Training Status

```bash
curl http://localhost:5002/api/train/status/{run_id}
```

### Architecture Progression

The AI system is designed to increase in complexity over time:

**Phase 1: Basic Feedforward Network**
- Simple 2-layer network
- ~50 input features (game state)
- 24 output probabilities (card selection)

**Phase 2: Intermediate**
- Separate networks for trump calling and card playing
- Increased hidden layer depth

**Phase 3: Advanced**
- LSTM/Transformer for game history
- Self-attention for opponent modeling
- Monte Carlo Tree Search integration

## 📊 Database Schema

The PostgreSQL database stores:
- **games** - Game metadata and final scores
- **hands** - Individual hand data within games
- **tricks** - Each trick played
- **moves** - Every card played with full context
- **game_states** - State snapshots for replay
- **ai_models** - Neural network weights and metrics
- **training_runs** - Genetic algorithm training data

## 🔧 Development

### Project Structure

```
EuchreBot/
├── .devcontainer/          # VS Code dev container config
├── services/
│   ├── euchre-api/        # Flask game API
│   ├── web-ui/            # Flask web interface
│   ├── cli/               # Terminal interface
│   └── ai-trainer/        # PyTorch training service
├── shared/
│   └── euchre_core/       # Core game engine (shared package)
├── database/
│   └── init.sql           # Database schema
├── docker-compose.yml     # Main orchestration
└── README.md
```

### Shared Core Package

The `euchre_core` package is shared across all services:

```python
from euchre_core import EuchreGame, PlayerType, Card, Suit

game = EuchreGame("game-001")
game.add_player("Alice", PlayerType.HUMAN)
game.start_new_hand()
```

### Running Tests

```bash
# Inside devcontainer or with Python environment
cd shared/euchre_core
pytest tests/
```

## 🎯 Game Rules (Implemented)

- Standard 4-player Euchre
- 24-card deck (9, 10, J, Q, K, A in each suit)
- Trump selection (2 rounds)
- Going alone
- Proper bower handling (right/left)
- Follow suit rules
- Scoring: 10 points to win
  - 3-4 tricks = 1 point
  - 5 tricks (march) = 2 points (4 if alone)
  - Euchred = 2 points to opponents

## 🔒 Security Notes

**Development Mode Only**
- This project is configured for localhost development
- Default passwords are used (`euchre_dev_pass`)
- All services bind to `127.0.0.1` only
- **Do not deploy to production without hardening!**

### Production Checklist
- [ ] Change all default passwords
- [ ] Use environment variables for secrets
- [ ] Enable HTTPS/TLS
- [ ] Add authentication/authorization
- [ ] Configure proper CORS policies
- [ ] Set up monitoring and logging
- [ ] Use production WSGI server (Gunicorn configured)

## 📈 Future Enhancements

- [ ] WebSocket support for real-time updates
- [ ] Tournament mode
- [ ] ELO rating system
- [ ] Advanced visualization (game replay with animations)
- [ ] Model comparison dashboard
- [ ] Export/import trained models
- [ ] Multi-GPU training support
- [ ] Distributed training across nodes

## 🤝 Contributing

Contributions are welcome! Areas of interest:
- Improving neural network architectures
- Better game state encoding
- UI/UX enhancements
- Additional AI training strategies
- Performance optimizations

## 📝 License

MIT License - See LICENSE file for details

## 🙏 Acknowledgments

- PyTorch team for the excellent deep learning framework
- Flask for the lightweight web framework
- The Euchre community for keeping this great game alive

## 📞 Contact

For questions or suggestions, please open an issue on GitHub.

---

**Happy Playing! 🃏**
