"""Training logic for continuous model training using Policy Gradient"""

import copy
from datetime import datetime
from model_manager import ModelManager
from networks.basic_nn import BasicEuchreNN
from networks.architecture_registry import ArchitectureRegistry
from training.policy_gradient import PolicyGradientTrainer


def run_continuous_training(
    run_id: str,
    population_size: int,
    games_per_pairing: int,
    database_url: str,
    num_workers: int,
    parallel_mode: str,
    use_cuda: bool,
    training_runs: dict,
    training_lock,
    cancellation_flags: dict,
    get_db_connection,
):
    """Run continuous policy gradient training until cancelled.

    Args:
        population_size: Ignored (kept for API compatibility). Single model training.
        games_per_pairing: Used as batch_size (games per gradient update).
    """
    try:
        with training_lock:
            training_runs[run_id]["status"] = "running"
            training_runs[run_id]["current_generation"] = 0
            cancellation_flags[run_id] = False

        # Create database entry for training run
        try:
            conn = get_db_connection()
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
            print(f"Error creating training run in DB: {e}")

        # Initialize model manager
        model_manager = ModelManager(database_url)

        # Try to load best existing model, otherwise create new
        model = None
        best_existing = model_manager.get_current_best_model()
        if best_existing is not None:
            _, model, existing_elo = best_existing
            print(f"Loaded best existing model (ELO: {existing_elo:.0f})")
        else:
            model = ArchitectureRegistry.create_model("basic", use_cuda=use_cuda)
            print("Created new model")

        # Training hyperparameters
        batch_size = max(games_per_pairing, 200)  # Use games_per_pairing but minimum 200
        learning_rate = 0.0001
        gamma = 0.99
        entropy_beta = 0.01
        exploration_rate = 0.1

        # Use semi-passive from the start:
        # - Opponents always pass trump (model team is always the caller)
        # - Opponents use frozen model for card play (trains against real defense)
        # This prevents both "never call" (Nash equilibrium) and "always call"
        # (trivially optimal against pure passive random opponents).
        from networks.critic_nn import EuchreCritic
        critic = EuchreCritic(use_cuda=use_cuda)

        trainer = PolicyGradientTrainer(
            model=model,
            learning_rate=learning_rate,
            gamma=gamma,
            entropy_beta=entropy_beta,
            exploration_rate=exploration_rate,
            use_cuda=use_cuda,
            self_play=True,  # enables opponent pool management for semi-passive
            opponent_update_interval=20,
            critic=critic,
            critic_lr=0.001,
        )

        print(f"\nPolicy Gradient Training Configuration:")
        print(f"  Batch Size:      {batch_size} games")
        print(f"  Learning Rate:   {learning_rate}")
        print(f"  Gamma:           {gamma}")
        print(f"  Entropy Beta:    {entropy_beta}")
        print(f"  Mode:            Semi-passive (opponents pass trump, use model for cards)")
        print(f"  Critic:          Enabled (Actor-Critic advantage computation)")
        print(f"  CUDA:            {use_cuda}")

        best_avg_reward = float("-inf")
        update = 0

        # Continuous training loop
        while not cancellation_flags.get(run_id, False):
            update += 1

            opponent_type = "semi_passive"
            phase_label = "Semi-Passive"

            print(f"\n{'='*60}")
            print(f"Update {update} (Policy Gradient - {phase_label})")
            print(f"{'='*60}")

            # Collect batch of episodes
            print(f"  Playing {batch_size} games...")
            episodes = []
            for game_num in range(batch_size):
                if cancellation_flags.get(run_id, False):
                    break
                episode = trainer.play_game(opponent_type=opponent_type)
                episodes.append(episode)

                if (game_num + 1) % 50 == 0:
                    print(f"    {game_num + 1}/{batch_size} games complete")

            if cancellation_flags.get(run_id, False):
                break

            # Train on batch
            print(f"  Training on batch...")
            stats = trainer.train_on_batch(episodes)

            # Print statistics
            print(f"\n  Update {update} Results:")
            print(f"    Loss:            {stats['loss']:.4f}")
            print(f"    Avg Reward:      {stats['avg_reward']:.4f}")
            print(f"    Running Reward:  {stats['running_reward']:.4f}")
            print(f"    Win Rate:        {stats['win_rate']*100:.1f}%")
            print(f"    Call Rate:       {stats['call_rate']*100:.1f}%")
            print(f"    Call Success:    {stats['call_success_rate']*100:.1f}%")
            print(f"    Entropy:         {stats.get('entropy', 0):.4f}")
            print(f"    Total Games:     {trainer.total_games}")

            # Update training state
            with training_lock:
                training_runs[run_id]["current_generation"] = update
                training_runs[run_id]["best_fitness"] = stats["running_reward"]
                training_runs[run_id]["avg_fitness"] = stats["avg_reward"]

            # Update database
            try:
                conn = get_db_connection()
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
                        f"Update {update}: WinRate={stats['win_rate']*100:.1f}%, "
                        f"CallRate={stats['call_rate']*100:.1f}%, "
                        f"Success={stats['call_success_rate']*100:.1f}%, "
                        f"Entropy={stats.get('entropy', 0):.4f}",
                    ),
                )

                conn.commit()
                cur.close()
                conn.close()
            except Exception as e:
                print(f"  Error updating database: {e}")

            # Save model every 10 updates
            if update % 10 == 0:
                estimated_elo = 1500 + (stats["avg_reward"] * 100)
                is_best = stats["avg_reward"] > best_avg_reward

                print(f"  Saving model (update {update})...")
                model_id = model_manager.save_model(
                    model,
                    f"PolicyGrad-Update{update}",
                    update,
                    stats["running_reward"],
                    run_id,
                    is_best=is_best,
                    elo_rating=estimated_elo,
                )

                if is_best and model_id:
                    best_avg_reward = stats["avg_reward"]
                    print(f"  NEW BEST MODEL! Avg Reward: {best_avg_reward:.4f}")
                    with training_lock:
                        training_runs[run_id]["best_model_id"] = model_id

        # Training was cancelled
        print(f"\nTraining cancelled at update {update}")

        # Save final model
        estimated_elo = 1500 + (trainer.avg_reward * 100)
        print(f"  Saving final model...")
        model_id = model_manager.save_model(
            model,
            f"PolicyGrad-Final-Update{update}",
            update,
            trainer.avg_reward,
            run_id,
            is_best=True,
            elo_rating=estimated_elo,
        )

        if model_id:
            with training_lock:
                training_runs[run_id]["best_model_id"] = model_id

        # Mark as completed
        with training_lock:
            training_runs[run_id]["status"] = "completed"

        # Update database
        try:
            conn = get_db_connection()
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
            print(f"Error completing training run: {e}")

        print(f"Training run {run_id} completed.")

    except Exception as e:
        print(f"Training error: {e}")
        import traceback

        traceback.print_exc()
        with training_lock:
            training_runs[run_id]["status"] = "failed"
            training_runs[run_id]["error"] = str(e)
