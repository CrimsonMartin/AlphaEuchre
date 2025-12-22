-- EuchreBot Database Schema

-- Games table - stores game metadata
CREATE TABLE IF NOT EXISTS games (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMP DEFAULT NOW(),
    completed_at TIMESTAMP,
    status VARCHAR(50) DEFAULT 'active',
    team1_score INTEGER DEFAULT 0,
    team2_score INTEGER DEFAULT 0,
    winner_team INTEGER,
    hand_count INTEGER DEFAULT 0
);

-- Players table - stores player information
CREATE TABLE IF NOT EXISTS players (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    game_id UUID REFERENCES games(id) ON DELETE CASCADE,
    position INTEGER NOT NULL CHECK (position >= 0 AND position <= 3),
    name VARCHAR(255) NOT NULL,
    player_type VARCHAR(50) NOT NULL,
    ai_model_id UUID,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE (game_id, position)
);

-- Hands table - stores individual hands within a game
CREATE TABLE IF NOT EXISTS hands (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    game_id UUID REFERENCES games(id) ON DELETE CASCADE,
    hand_number INTEGER NOT NULL,
    dealer_position INTEGER NOT NULL,
    trump_suit VARCHAR(1),
    trump_caller_position INTEGER,
    going_alone BOOLEAN DEFAULT FALSE,
    alone_player_position INTEGER,
    team1_tricks INTEGER DEFAULT 0,
    team2_tricks INTEGER DEFAULT 0,
    winning_team INTEGER,
    points_awarded INTEGER,
    started_at TIMESTAMP DEFAULT NOW(),
    completed_at TIMESTAMP,
    UNIQUE(game_id, hand_number)
);

-- Tricks table - stores each trick played
CREATE TABLE IF NOT EXISTS tricks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    hand_id UUID REFERENCES hands(id) ON DELETE CASCADE,
    trick_number INTEGER NOT NULL,
    lead_position INTEGER NOT NULL,
    winner_position INTEGER,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(hand_id, trick_number)
);

-- Moves table - stores each card played
CREATE TABLE IF NOT EXISTS moves (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    trick_id UUID REFERENCES tricks(id) ON DELETE CASCADE,
    player_position INTEGER NOT NULL,
    card VARCHAR(3) NOT NULL,
    play_order INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Game state snapshots - for replay and training
CREATE TABLE IF NOT EXISTS game_states (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    game_id UUID REFERENCES games(id) ON DELETE CASCADE,
    hand_id UUID REFERENCES hands(id) ON DELETE CASCADE,
    trick_id UUID REFERENCES tricks(id) ON DELETE CASCADE,
    state_json JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

-- AI Models table - stores neural network models
CREATE TABLE IF NOT EXISTS ai_models (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    version VARCHAR(50) NOT NULL,
    architecture VARCHAR(100) NOT NULL,
    generation INTEGER DEFAULT 1,
    parent_model_id UUID REFERENCES ai_models(id),
    training_run_id UUID,
    model_weights BYTEA,
    model_path VARCHAR(500),
    performance_metrics JSONB,
    elo_rating FLOAT DEFAULT 1500,
    wins INTEGER DEFAULT 0,
    losses INTEGER DEFAULT 0,
    games_played INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW(),
    active BOOLEAN DEFAULT TRUE,
    is_best_overall BOOLEAN DEFAULT FALSE
);

-- Training runs table - tracks genetic algorithm runs
CREATE TABLE IF NOT EXISTS training_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255),
    generation_count INTEGER DEFAULT 0,
    population_size INTEGER NOT NULL,
    mutation_rate FLOAT,
    crossover_rate FLOAT,
    elite_size INTEGER,
    best_fitness FLOAT DEFAULT 1500,
    avg_fitness FLOAT DEFAULT 1500,
    config JSONB,
    status VARCHAR(50) DEFAULT 'running',
    started_at TIMESTAMP DEFAULT NOW(),
    completed_at TIMESTAMP
);

-- Training games - games played during training
CREATE TABLE IF NOT EXISTS training_games (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    training_run_id UUID REFERENCES training_runs(id) ON DELETE CASCADE,
    game_id UUID REFERENCES games(id) ON DELETE CASCADE,
    generation INTEGER NOT NULL,
    model_ids UUID[] NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Training logs - generation-by-generation training history
CREATE TABLE IF NOT EXISTS training_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    training_run_id UUID REFERENCES training_runs(id) ON DELETE CASCADE,
    generation INTEGER NOT NULL,
    best_fitness FLOAT NOT NULL,
    avg_fitness FLOAT NOT NULL,
    message TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(training_run_id, generation)
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_games_status ON games(status);
CREATE INDEX IF NOT EXISTS idx_games_created_at ON games(created_at);
CREATE INDEX IF NOT EXISTS idx_players_game_id ON players(game_id);
CREATE INDEX IF NOT EXISTS idx_hands_game_id ON hands(game_id);
CREATE INDEX IF NOT EXISTS idx_tricks_hand_id ON tricks(hand_id);
CREATE INDEX IF NOT EXISTS idx_moves_trick_id ON moves(trick_id);
CREATE INDEX IF NOT EXISTS idx_game_states_game_id ON game_states(game_id);
CREATE INDEX IF NOT EXISTS idx_ai_models_active ON ai_models(active);
CREATE INDEX IF NOT EXISTS idx_ai_models_generation ON ai_models(generation);
CREATE INDEX IF NOT EXISTS idx_training_runs_status ON training_runs(status);
CREATE INDEX IF NOT EXISTS idx_training_games_run_id ON training_games(training_run_id);
CREATE INDEX IF NOT EXISTS idx_training_logs_run_id ON training_logs(training_run_id);
