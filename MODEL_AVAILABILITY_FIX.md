# Model Availability Fix - Trained Models in UI Dropdown

## Problem Summary

Trained AI models were not appearing in the dropdown menu on the "New Game" page, preventing users from selecting and playing against trained neural network models.

## Root Cause Analysis

After investigating the complete flow from frontend to database, the issue was identified:

### The Architecture (Working Correctly)
1. **Frontend** (`new_game.html`) - Fetches models from `/api/models` endpoint
2. **Web-UI Service** - Proxies request to `ai-trainer:5003/api/models`
3. **AI Trainer Service** - Queries database: `SELECT ... FROM ai_models WHERE active = true`
4. **Database** - Returns saved models

### The Actual Problem
**No models were being saved to the database** because:
- Models were only auto-saved every **10 generations**
- Existing training runs had only reached generations 4-5
- Therefore, no models existed in the `ai_models` table

### Verification
```sql
SELECT COUNT(*) FROM ai_models WHERE active = true;
-- Result: 0 rows
```

## Solution Implemented

### 1. Reduced Auto-Save Frequency
**File:** `services/ai-trainer/app.py`

**Change:** Modified auto-save interval from every 10 generations to every 5 generations, AND added save at generation 1.

```python
# Before:
if generation % 10 == 0 and ga.global_best_model is not None:

# After:
if (generation == 1 or generation % 5 == 0) and ga.global_best_model is not None:
```

**Benefits:**
- Models appear much faster (after just 1 generation instead of 10)
- More frequent checkpoints for model recovery
- Users can start playing against trained models sooner

### 2. Improved Error Visibility
**File:** `services/web-ui/app/templates/new_game.html`

**Changes:**
- Added informative message when no models are available: `"(No trained models yet - run training first)"`
- Added error message when API call fails: `"(Error loading models - check AI trainer service)"`
- Better user feedback instead of silent failures

**Before:** Empty dropdown with no explanation
**After:** Clear messages explaining why models aren't available

## How to Verify the Fix

### 1. Check if models exist in database:
```bash
docker-compose exec -T postgres psql -U euchre -d euchrebot -c \
  "SELECT id, name, generation, elo_rating FROM ai_models WHERE active = true;"
```

### 2. Start a new training run:
- Navigate to http://localhost:5001/training
- Click "Start Training"
- Wait for generation 1 to complete (models will auto-save)

### 3. Check the New Game page:
- Navigate to http://localhost:5001/new-game
- Check Player 2, 3, and 4 dropdowns
- Trained models should appear with their ELO ratings

### 4. Verify model appears in dropdown:
The dropdown should show:
```
Random AI
Human
--- Trained Models ---
AutoSave-Gen1 (ELO: 1500)
AutoSave-Gen5 (ELO: 1650)
...
```

## Additional Notes

### Model Save Points
Models are now saved at:
1. **Generation 1** - Initial best model
2. **Every 5 generations** - Regular checkpoints (Gen 5, 10, 15, 20, etc.)
3. **When training is cancelled** - Final best model

### Model Naming Convention
- Auto-saved models: `AutoSave-Gen{N}` (e.g., "AutoSave-Gen5")
- Final models: `FinalBest-Gen{N}` (e.g., "FinalBest-Gen23")

### Playing Against Trained Models
Once models appear in the dropdown:
1. Select a trained model for any AI player (Players 2, 3, or 4)
2. The model ID is passed as `neural_net_ai:{model_id}`
3. During gameplay, the `ai-trainer` service's `/api/models/{model_id}/predict` endpoint is called
4. The neural network makes intelligent card-playing decisions

## Services Restarted
- `ai-trainer` - To apply the new auto-save frequency
- `web-ui` - To apply the improved error messages

## Testing Checklist
- [x] Verified no models exist in database initially
- [x] Confirmed training runs were at generation 4-5 (below old threshold of 10)
- [x] Updated auto-save logic to save at gen 1 and every 5 generations
- [x] Added user-friendly error messages to UI
- [x] Restarted affected services
- [ ] Start new training run and verify model saves at generation 1
- [ ] Verify model appears in dropdown on New Game page
- [ ] Test playing a game against a trained model

## Future Improvements

1. **Manual Save Button** - Allow users to manually save the current best model at any time
2. **Model Management UI** - View, rename, and delete saved models
3. **Model Comparison** - Compare performance metrics between different models
4. **Export/Import** - Download and share trained models
