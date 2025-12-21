# AI Training System - Round-Robin Tournament Architecture

## Overview

The AI training system has been redesigned to use a **round-robin tournament approach** where models compete in teams. This provides more robust evaluation of model performance and better genetic algorithm evolution.

## Key Features

### 1. Round-Robin Tournament System

Instead of random matchups, models now compete in structured tournaments:

- **Group Size**: 4 models per tournament group
- **Team Pairings**: Each model plays with every other model as a teammate
- **Games per Pairing**: 10 games for statistical significance

#### Tournament Structure

For each group of 4 models (A, B, C, D):

```
Round 1: Team (A+B) vs Team (C+D) → 10 games
Round 2: Team (A+C) vs Team (B+D) → 10 games  
Round 3: Team (A+D) vs Team (B+C) → 10 games
```

**Total**: 30 games per evaluation cycle, each model plays in all 30 games

### 2. Enhanced Fitness Scoring

Models are evaluated using a comprehensive fitness metric:

```python
fitness = (win_rate * 0.8) + (normalized_point_differential * 0.2)
```

Where:
- **Win Rate** (80% weight): Games won / Games played
- **Point Differential** (20% weight): (Points for - Points against) / (Games * 10)

This rewards both winning games and performing well even in losses.

### 3. Model Persistence & Seeding

#### Model Manager (`model_manager.py`)

New module that handles:
- **Saving models**: Stores weights to `/models/` directory with metadata in database
- **Loading models**: Retrieves models by ID from filesystem
- **Best model tracking**: Queries top performers across all training runs
- **Population seeding**: Initializes new populations from previous best models

#### Seeding Configuration

- **Seed Percentage**: 25% of population from best previous models
- **New Models**: 75% randomly initialized
- **Selection**: Top models ranked by fitness score

### 4. Database Schema Updates

Added to `ai_models` table:
```sql
is_best_overall BOOLEAN DEFAULT FALSE
```

This flag marks the best models from each training run for easy retrieval.

## Usage

### Starting a Training Run

#### Via Web UI

1. Navigate to `/training`
2. Configure parameters:
   - **Population Size**: Number of models (must be divisible by 4 for optimal performance)
   - **Generations**: Number of evolution cycles
   - **Seed from Best Models**: Check to use 25% seeding from previous runs
3. Click "Start Training"

#### Via API

```bash
curl -X POST http://localhost:5001/api/train/start \
  -H "Content-Type: application/json" \
  -d '{
    "population_size": 20,
    "generations": 10,
    "use_seeding": true
  }'
```

### Monitoring Training

The web UI provides real-time updates:
- Current generation progress
- Best fitness score
- Average fitness score
- Training logs

## Architecture Components

### Files Modified/Created

1. **`services/ai-trainer/model_manager.py`** (NEW)
   - Model persistence and retrieval
   - Population seeding logic

2. **`services/ai-trainer/genetic/genetic_algorithm.py`** (UPDATED)
   - Round-robin tournament implementation
   - Team-based game playing
   - Enhanced fitness evaluation

3. **`services/ai-trainer/app.py`** (UPDATED)
   - Integration with ModelManager
   - Seeding parameter support
   - Best model saving with `is_best=True` flag

4. **`services/web-ui/app/templates/training.html`** (UPDATED)
   - Seeding checkbox UI
   - Help text for seeding option

5. **`services/web-ui/app/routes/main_routes.py`** (UPDATED)
   - Handle seeding checkbox from form
   - Pass to AI trainer API

6. **`database/init.sql`** (UPDATED)
   - Added `is_best_overall` column

## Training Flow

```
1. User starts training (optionally with seeding)
   ↓
2. ModelManager loads best models if seeding enabled
   ↓
3. GeneticAlgorithm initializes population
   - 25% from loaded best models
   - 75% new random models
   ↓
4. For each generation:
   a. Divide population into groups of 4
   b. Run round-robin tournament for each group
   c. Calculate fitness scores
   d. Select elite models
   e. Generate offspring via crossover + mutation
   ↓
5. Save best model with is_best_overall=true
   ↓
6. Model available for future seeding
```

## Performance Considerations

### Games per Generation

For a population of 20 models:
- **Groups**: 5 groups of 4 models
- **Pairings per group**: 3 pairings
- **Games per pairing**: 10 games
- **Total games**: 5 × 3 × 10 = **150 games per generation**

### Training Time Estimates

Approximate times (depends on hardware):
- **Single game**: ~1-2 seconds
- **Generation (pop=20)**: ~3-5 minutes
- **10 generations**: ~30-50 minutes

## Best Practices

1. **Population Size**: Use multiples of 4 (e.g., 20, 24, 28) for optimal tournament grouping

2. **Seeding**: 
   - First run: Don't use seeding (no previous models)
   - Subsequent runs: Enable seeding to build on previous progress

3. **Generations**:
   - Quick test: 5-10 generations
   - Serious training: 20-50 generations
   - Long-term evolution: 100+ generations

4. **Model Storage**:
   - Models saved to `/models/` directory
   - Ensure sufficient disk space for long training runs
   - Each model ~100KB

## Future Enhancements

Potential improvements:
- [ ] Configurable seeding percentage
- [ ] Multi-generational lineage tracking
- [ ] Tournament bracket visualization
- [ ] Model comparison tools
- [ ] Automated hyperparameter tuning
- [ ] Distributed training across multiple nodes

## Troubleshooting

### Models not seeding
- Check database for models with `is_best_overall=true`
- Verify `/models/` directory exists and is writable
- Check logs for ModelManager errors

### Training runs slowly
- Reduce population size
- Reduce games per pairing (modify `games_per_pairing` in code)
- Check system resources (CPU/memory)

### Database errors
- Ensure PostgreSQL is running
- Verify database schema is up to date
- Check connection string in environment variables

## Technical Details

### Genetic Algorithm Parameters

```python
GeneticAlgorithm(
    population_size=20,      # Total models in population
    mutation_rate=0.1,       # 10% chance of mutation per weight
    crossover_rate=0.7,      # 70% chance of crossover
    elite_size=2,            # Top 2 models preserved each generation
    games_per_pairing=10     # Games per team pairing
)
```

### Neural Network Architecture

```
Input Layer:  65 features (game state encoding)
Hidden Layer: 64 neurons (ReLU activation)
Hidden Layer: 64 neurons (ReLU activation)
Output Layer: 24 neurons (card probabilities, Softmax)
```

## References

- Genetic Algorithms: Holland, J. H. (1992). "Adaptation in Natural and Artificial Systems"
- Neural Networks: Goodfellow, I., et al. (2016). "Deep Learning"
- Euchre Strategy: Various online resources and game theory papers
