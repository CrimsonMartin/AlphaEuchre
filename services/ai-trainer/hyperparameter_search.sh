#!/bin/bash
#
# Hyperparameter Search Script for Policy Gradient Training
# Searches for optimal learning rate, batch size, and exploration rate
#
# Usage: bash hyperparameter_search.sh
#

set -e

# Configuration
CONTAINER_NAME="euchre-ai-trainer"
NUM_UPDATES=50  # Quick test runs
ARCHITECTURE="cnn"
# ARCHITECTURE="basic"
# ARCHITECTURE="transformer"

# Output file
RESULTS_FILE="hyperparameter_results.txt"
echo "Hyperparameter Search Results - $(date)" > $RESULTS_FILE
echo "========================================" >> $RESULTS_FILE
echo "" >> $RESULTS_FILE

# Color codes for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting Hyperparameter Search${NC}"
echo "Results will be saved to: $RESULTS_FILE"
echo ""

# Counter for experiments
experiment=0

# Function to run training and extract final win rate
run_experiment() {
    local batch_size=$1
    local learning_rate=$2
    local exploration_rate=$3
    local gamma=$4
    
    experiment=$((experiment + 1))
    
    echo -e "${YELLOW}Experiment $experiment:${NC}"
    echo "  Batch Size: $batch_size"
    echo "  Learning Rate: $learning_rate"
    echo "  Exploration Rate: $exploration_rate"
    echo "  Gamma: $gamma"
    echo ""
    
    # Run training and capture output
    output=$(docker exec -it $CONTAINER_NAME python train_gradient.py \
        --batch-size $batch_size \
        --num-updates $NUM_UPDATES \
        --learning-rate $learning_rate \
        --exploration-rate $exploration_rate \
        --gamma $gamma \
        --architecture $ARCHITECTURE \
        --save-every 999 \
        2>&1 || true)
    
    # Extract final win rate (look for "Win Rate:" in final statistics)
    win_rate=$(echo "$output" | grep "Win Rate:" | tail -1 | awk '{print $3}')
    
    # Extract running reward
    running_reward=$(echo "$output" | grep "Running Reward:" | tail -1 | awk '{print $3}')
    
    # If extraction failed, set to N/A
    if [ -z "$win_rate" ]; then
        win_rate="N/A"
    fi
    if [ -z "$running_reward" ]; then
        running_reward="N/A"
    fi
    
    echo -e "${GREEN}  Result: Win Rate = $win_rate, Running Reward = $running_reward${NC}"
    echo ""
    
    # Log to file
    echo "Experiment $experiment:" >> $RESULTS_FILE
    echo "  Batch Size: $batch_size" >> $RESULTS_FILE
    echo "  Learning Rate: $learning_rate" >> $RESULTS_FILE
    echo "  Exploration Rate: $exploration_rate" >> $RESULTS_FILE
    echo "  Gamma: $gamma" >> $RESULTS_FILE
    echo "  Win Rate: $win_rate" >> $RESULTS_FILE
    echo "  Running Reward: $running_reward" >> $RESULTS_FILE
    echo "" >> $RESULTS_FILE
}

# Grid Search
echo -e "${GREEN}Phase 1: Testing Different Learning Rates${NC}"
echo "=========================================="
echo ""

# Test learning rates (keeping other params constant)
for lr in 0.0001 0.00005 0.00001; do
    run_experiment 100 $lr 0.1 0.95
done

echo -e "${GREEN}Phase 2: Testing Different Batch Sizes${NC}"
echo "=========================================="
echo ""

# Test batch sizes (using best LR from phase 1 - adjust if needed)
for bs in 50 100 200; do
    run_experiment $bs 0.00005 0.1 0.95
done

echo -e "${GREEN}Phase 3: Testing Different Exploration Rates${NC}"
echo "=========================================="
echo ""

# Test exploration rates
for er in 0.05 0.1 0.15; do
    run_experiment 100 0.00005 $er 0.95
done

echo -e "${GREEN}Phase 4: Testing Different Gamma Values${NC}"
echo "=========================================="
echo ""

# Test gamma values
for g in 0.90 0.95 0.99; do
    run_experiment 100 0.00005 0.1 $g
done

echo -e "${GREEN}Phase 5: Testing Best Combinations${NC}"
echo "=========================================="
echo ""

# Test some promising combinations
run_experiment 200 0.00005 0.05 0.95
run_experiment 200 0.00001 0.05 0.95
run_experiment 150 0.00005 0.05 0.99

echo ""
echo -e "${GREEN}Hyperparameter Search Complete!${NC}"
echo "Results saved to: $RESULTS_FILE"
echo ""
echo "To view results:"
echo "  cat $RESULTS_FILE"
echo ""
echo "To find best configuration:"
echo "  grep -A 5 'Win Rate:' $RESULTS_FILE | sort -t ':' -k 2 -rn | head -20"
