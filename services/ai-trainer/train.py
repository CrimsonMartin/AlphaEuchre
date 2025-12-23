#!/usr/bin/env python3
"""
Standalone training script for interactive Python execution.

Usage:
    python train.py --population-size 20 --games-per-pairing 100

    # With custom database URL:
    DATABASE_URL="postgresql://..." python train.py

    # Run inside Docker container:
    docker exec -it euchrebot-ai-trainer-1 python train.py

    # Stop training gracefully:
    Press Ctrl+C to finish current generation and save best model
"""

import argparse
import signal
import sys
import os
import uuid
import multiprocessing
import threading
import psycopg2

# Add paths for imports
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "shared"))

import torch
from training.trainer import run_continuous_training

# Global flag for graceful shutdown
shutdown_requested = False


def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully"""
    global shutdown_requested
    print("\n\n⚠️  Shutdown requested. Finishing current generation...")
    print("    (Press Ctrl+C again to force quit)")
    shutdown_requested = True


def main():
    parser = argparse.ArgumentParser(
        description="Train EuchreBot AI models using genetic algorithm",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic training with default settings
  python train.py
  
  # Larger population for better diversity
  python train.py --population-size 40
  
  # More games for better evaluation
  python train.py --games-per-pairing 200
  
  # Run inside Docker container
  docker exec -it euchrebot-ai-trainer-1 python train.py --population-size 20
  
  # Custom database connection
  DATABASE_URL="postgresql://user:pass@host:5432/db" python train.py
        """,
    )

    parser.add_argument(
        "--population-size",
        type=int,
        default=20,
        help="Population size - number of models per generation (default: 20)",
    )

    parser.add_argument(
        "--games-per-pairing",
        type=int,
        default=100,
        help="Number of games each model pair plays for evaluation (default: 100)",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=max(1, multiprocessing.cpu_count() - 1),
        help=f"Number of parallel workers (default: {max(1, multiprocessing.cpu_count() - 1)})",
    )

    parser.add_argument(
        "--parallel-mode",
        choices=["thread", "process", "sequential"],
        default="thread",
        help="Parallelization mode: thread (default), process, or sequential",
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
    print("🤖 EuchreBot AI Training - Interactive Mode")
    print("=" * 70)
    print(f"Population Size:    {args.population_size}")
    print(f"Games per Pairing:  {args.games_per_pairing}")
    print(f"Num Workers:        {args.num_workers}")
    print(f"Parallel Mode:      {args.parallel_mode}")
    print(f"CUDA Available:     {torch.cuda.is_available()}")
    print(f"CUDA Enabled:       {use_cuda}")
    if use_cuda and torch.cuda.is_available():
        print(f"GPU Device:         {torch.cuda.get_device_name(0)}")

    # Hide credentials in database URL for display
    db_display = database_url.split("@")[-1] if "@" in database_url else database_url
    print(f"Database:           {db_display}")
    print("=" * 70)
    print("Training will run continuously until you press Ctrl+C")
    print("Best models are auto-saved every 5 generations")
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
        print("  docker exec -it euchrebot-ai-trainer-1 python train.py")
        sys.exit(1)

    # Create training run ID
    run_id = str(uuid.uuid4())
    print(f"Training Run ID: {run_id}")
    print()

    # Shared state (simplified for standalone use)
    training_runs = {
        run_id: {
            "status": "starting",
            "population_size": args.population_size,
            "continuous": True,
            "current_generation": 0,
            "best_fitness": 1500.0,
            "avg_fitness": 1500.0,
        }
    }
    training_lock = threading.Lock()
    cancellation_flags = {run_id: False}

    def get_db_connection():
        """Database connection factory"""
        return psycopg2.connect(database_url)

    # Monitor for shutdown in background
    def monitor_shutdown():
        """Background thread to monitor shutdown flag"""
        global shutdown_requested
        while not shutdown_requested:
            import time

            time.sleep(0.5)
        cancellation_flags[run_id] = True

    monitor_thread = threading.Thread(target=monitor_shutdown, daemon=True)
    monitor_thread.start()

    # Run training (blocking call)
    try:
        run_continuous_training(
            run_id=run_id,
            population_size=args.population_size,
            games_per_pairing=args.games_per_pairing,
            database_url=database_url,
            num_workers=args.num_workers,
            parallel_mode=args.parallel_mode,
            use_cuda=use_cuda,
            training_runs=training_runs,
            training_lock=training_lock,
            cancellation_flags=cancellation_flags,
            get_db_connection=get_db_connection,
        )
    except KeyboardInterrupt:
        print("\n\n⚠️  Force quit detected. Training may not have saved properly.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Training error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    # Print final status
    print("\n" + "=" * 70)
    print("✅ Training Complete!")
    print("=" * 70)
    final_status = training_runs[run_id]
    print(f"Status:             {final_status.get('status', 'unknown')}")
    print(f"Generations:        {final_status.get('current_generation', 0)}")
    print(f"Best ELO:           {final_status.get('best_fitness', 0):.0f}")
    print(f"Average ELO:        {final_status.get('avg_fitness', 0):.0f}")
    if "best_model_id" in final_status:
        print(f"Best Model ID:      {final_status['best_model_id']}")
    print("=" * 70)
    print("\nYou can analyze the trained models at:")
    print("  http://localhost:5001/model-strategy")
    print()


if __name__ == "__main__":
    main()
