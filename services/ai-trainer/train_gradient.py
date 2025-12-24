#!/usr/bin/env python3
"""
Policy Gradient Training Script for Euchre AI
Uses REINFORCE algorithm with reward shaping

Usage:
    python train_gradient.py --batch-size 100 --learning-rate 0.0003

    # Resume from existing model:
    python train_gradient.py --load-model <model-id> --batch-size 100

    # Run inside Docker container:
    docker exec -it euchrebot-ai-trainer-1 python train_gradient.py

    # Stop training gracefully:
    Press Ctrl+C to finish current batch and save model
"""

import argparse
import signal
import sys
import os
import uuid
import psycopg2
from datetime import datetime

# Add paths for imports
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "shared"))

import torch
from networks.basic_nn import BasicEuchreNN
from networks.architecture_registry import ArchitectureRegistry
from training.policy_gradient import PolicyGradientTrainer
from model_manager import ModelManager

# Global flag for graceful shutdown
shutdown_requested = False


def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully"""
    global shutdown_requested
    print("\n\n⚠️  Shutdown requested. Finishing current batch...")
    print("    (Press Ctrl+C again to force quit)")
    shutdown_requested = True


def main():
    parser = argparse.ArgumentParser(
        description="Train EuchreBot AI using Policy Gradient (REINFORCE)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic training with default settings
  python train_gradient.py
  
  # Custom hyperparameters
  python train_gradient.py --batch-size 100 --learning-rate 0.0003
  
  # Resume from existing model
  python train_gradient.py --load-model <model-id>
  
  # Run inside Docker container
  docker exec -it euchrebot-ai-trainer-1 python train_gradient.py
  
  # Custom database connection
  DATABASE_URL="postgresql://user:pass@host:5432/db" python train_gradient.py
        """,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Number of games per gradient update (default: 50)",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.0001,
        help="Learning rate for Adam optimizer (default: 0.0001)",
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor for future rewards (default: 0.99)",
    )

    parser.add_argument(
        "--entropy-beta",
        type=float,
        default=0.01,
        help="Entropy bonus coefficient for exploration (default: 0.01)",
    )

    parser.add_argument(
        "--exploration-rate",
        type=float,
        default=0.1,
        help="Epsilon-greedy exploration rate (default: 0.1)",
    )

    parser.add_argument(
        "--num-updates",
        type=int,
        default=1000,
        help="Number of batch updates to perform (default: 1000)",
    )

    parser.add_argument(
        "--save-every",
        type=int,
        default=50,
        help="Save model every N updates (default: 50)",
    )

    parser.add_argument(
        "--load-model",
        type=str,
        default=None,
        help="Model ID to resume training from (optional)",
    )

    parser.add_argument(
        "--architecture",
        type=str,
        choices=["basic", "cnn", "transformer"],
        default="basic",
        help="Neural network architecture to use (default: basic)",
    )

    parser.add_argument(
        "--no-cuda", action="store_true", help="Disable CUDA even if available"
    )

    parser.add_argument(
        "--database-url",
        type=str,
        default=os.getenv(
            "DATABASE_URL",
            "postgresql://euchre:euchre_dev_pass@localhost:5432/euchrebot",
        ),
        help="Database connection URL (default: from DATABASE_URL env var)",
    )

    args = parser.parse_args()

    # Setup signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Configuration
    database_url = args.database_url
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    # Print configuration
    print("=" * 70)
    print("🎯 EuchreBot AI Training - Policy Gradient (REINFORCE)")
    print("=" * 70)
    print(f"Batch Size:         {args.batch_size} games")
    print(f"Learning Rate:      {args.learning_rate}")
    print(f"Discount (γ):       {args.gamma}")
    print(f"Entropy Beta:       {args.entropy_beta}")
    print(f"Exploration (ε):    {args.exploration_rate}")
    print(f"Num Updates:        {args.num_updates}")
    print(f"Save Every:         {args.save_every} updates")
    print(f"CUDA Available:     {torch.cuda.is_available()}")
    print(f"CUDA Enabled:       {use_cuda}")
    if use_cuda and torch.cuda.is_available():
        print(f"GPU Device:         {torch.cuda.get_device_name(0)}")

    # Hide credentials in database URL for display
    db_display = database_url.split("@")[-1] if "@" in database_url else database_url
    print(f"Database:           {db_display}")
    print("=" * 70)
    print()

    # Test database connection
    try:
        print("Testing database connection...")
        conn = psycopg2.connect(database_url)
        conn.close()
        print("✓ Database connection successful")
        print()
    except Exception as e:
        print(f"✗ Database connection failed: {e}")
        print("\nPlease check your DATABASE_URL and ensure the database is running.")
        print("If using Docker, make sure you're running this inside the container:")
        print("  docker exec -it euchrebot-ai-trainer-1 python train_gradient.py")
        sys.exit(1)

    # Initialize model manager
    model_manager = ModelManager(database_url)

    # Load or create model
    if args.load_model:
        print(f"Loading model {args.load_model}...")
        model = model_manager.load_model(args.load_model)
        if model is None:
            print(f"✗ Failed to load model {args.load_model}")
            sys.exit(1)

        # Get architecture info
        arch_type = ArchitectureRegistry.get_architecture_type(model)
        arch_info = ArchitectureRegistry.get_architecture_info(arch_type)
        print(f"✓ Model loaded successfully")
        print(f"  Architecture: {arch_info['name']} ({arch_info['description']})")
        print()
    else:
        print(f"Creating new {args.architecture} model...")
        model = ArchitectureRegistry.create_model(args.architecture, use_cuda=use_cuda)

        # Get architecture info
        arch_info = ArchitectureRegistry.get_architecture_info(args.architecture)
        print(f"✓ New model created")
        print(f"  Architecture: {arch_info['name']} ({arch_info['description']})")
        print()

    # Create training run ID
    run_id = str(uuid.uuid4())
    print(f"Training Run ID: {run_id}")
    print()

    # Create database entry for training run
    try:
        conn = psycopg2.connect(database_url)
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO training_runs (id, name, generation_count, population_size, 
                                     mutation_rate, crossover_rate, elite_size, status)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """,
            (
                run_id,
                f"Policy Gradient {run_id[:8]}",
                0,
                1,  # Single model
                0.0,  # N/A for policy gradient
                0.0,  # N/A for policy gradient
                1,  # Single model
                "running",
            ),
        )
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Warning: Could not create training run in DB: {e}")

    # Initialize trainer
    trainer = PolicyGradientTrainer(
        model=model,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        entropy_beta=args.entropy_beta,
        exploration_rate=args.exploration_rate,
        use_cuda=use_cuda,
    )

    print("=" * 70)
    print("Starting training...")
    print("=" * 70)
    print()

    best_avg_reward = float("-inf")
    best_model_id = None

    # Training loop
    try:
        for update in range(1, args.num_updates + 1):
            if shutdown_requested:
                break

            print(f"Update {update}/{args.num_updates}")
            print("-" * 70)

            # Collect batch of episodes
            print(f"  Playing {args.batch_size} games...")
            episodes = []
            for game_num in range(args.batch_size):
                if shutdown_requested:
                    break
                episode = trainer.play_game()
                episodes.append(episode)

                # Progress indicator
                if (game_num + 1) % 10 == 0:
                    print(
                        f"\r  {game_num + 1}/{args.batch_size} games complete",
                        end="",
                        flush=True,
                    )

            if shutdown_requested:
                break

            # Train on batch
            print(f"  Training on batch...")
            stats = trainer.train_on_batch(episodes)

            # Print statistics
            print(f"\n  Update {update} Results:")
            print(f"    Loss:            {stats['loss']:.4f}")
            print(f"    Avg Reward:      {stats['avg_reward']:.4f}")
            print(f"    Win Rate:        {stats['win_rate']:.2%}")
            print(f"    Running Reward:  {stats['running_reward']:.4f}")
            print(f"    Total Games:     {trainer.total_games}")
            print(f"    Total Wins:      {trainer.total_wins}")
            print()

            # Update database
            try:
                conn = psycopg2.connect(database_url)
                cur = conn.cursor()
                cur.execute(
                    """
                    UPDATE training_runs 
                    SET generation_count = %s, best_fitness = %s, avg_fitness = %s
                    WHERE id = %s
                """,
                    (update, stats["running_reward"], stats["avg_reward"], run_id),
                )

                # Insert training log entry
                cur.execute(
                    """
                    INSERT INTO training_logs (training_run_id, generation, best_fitness, avg_fitness, message)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (training_run_id, generation) DO NOTHING
                """,
                    (
                        run_id,
                        update,
                        stats["running_reward"],
                        stats["avg_reward"],
                        f"Update {update}: Reward = {stats['avg_reward']:.4f}, Win Rate = {stats['win_rate']:.2%}",
                    ),
                )

                conn.commit()
                cur.close()
                conn.close()
            except Exception as e:
                print(f"  Warning: Could not update database: {e}")

            # Save model periodically
            if update % args.save_every == 0 or update == args.num_updates:
                print(f"  💾 Saving model (update {update})...")

                # Estimate ELO based on win rate (rough approximation)
                # Win rate of 50% = 1500 ELO, each 10% = ~100 ELO
                estimated_elo = 1500 + (stats["win_rate"] - 0.5) * 1000

                model_id = model_manager.save_model(
                    model,
                    f"PolicyGrad-Update{update}",
                    update,
                    stats["running_reward"],
                    run_id,
                    is_best=(stats["avg_reward"] > best_avg_reward),
                    elo_rating=estimated_elo,
                )

                if model_id and stats["avg_reward"] > best_avg_reward:
                    best_avg_reward = stats["avg_reward"]
                    best_model_id = model_id
                    print(f"  🏆 NEW BEST MODEL! Avg Reward: {best_avg_reward:.4f}")

                print()

    except KeyboardInterrupt:
        print("\n\n⚠️  Force quit detected. Saving current model...")
    except Exception as e:
        print(f"\n\n❌ Training error: {e}")
        import traceback

        traceback.print_exc()

    # Save final model
    print("\n" + "=" * 70)
    print("Training Complete - Saving Final Model")
    print("=" * 70)

    estimated_elo = (
        1500 + (trainer.total_wins / max(1, trainer.total_games) - 0.5) * 1000
    )

    final_model_id = model_manager.save_model(
        model,
        f"PolicyGrad-Final-Update{update}",
        update,
        trainer.avg_reward,
        run_id,
        is_best=True,
        elo_rating=estimated_elo,
    )

    # Update database status
    try:
        conn = psycopg2.connect(database_url)
        cur = conn.cursor()
        cur.execute(
            """
            UPDATE training_runs 
            SET status = %s, completed_at = %s
            WHERE id = %s
        """,
            ("completed", datetime.now(), run_id),
        )
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Warning: Could not update training run status: {e}")

    # Print final statistics
    print()
    print("Final Statistics:")
    print(f"  Total Games:      {trainer.total_games}")
    print(f"  Total Wins:       {trainer.total_wins}")
    print(f"  Win Rate:         {trainer.total_wins / max(1, trainer.total_games):.2%}")
    print(f"  Avg Reward:       {trainer.avg_reward:.4f}")
    if trainer.running_reward is not None:
        print(f"  Running Reward:   {trainer.running_reward:.4f}")
    print(f"  Estimated ELO:    {estimated_elo:.0f}")
    if final_model_id:
        print(f"  Final Model ID:   {final_model_id}")
    if best_model_id:
        print(f"  Best Model ID:    {best_model_id}")
    print("=" * 70)
    print()
    print("You can analyze the trained model at:")
    print("  http://localhost:5001/model-strategy")
    print()


if __name__ == "__main__":
    main()
