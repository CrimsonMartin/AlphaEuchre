"""Training logic for continuous model training"""

import copy
from datetime import datetime
from model_manager import ModelManager
from genetic.genetic_algorithm import GeneticAlgorithm


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
    """Run continuous training until cancelled"""
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
                    f"Continuous Training {run_id[:8]}",
                    0,
                    population_size,
                    0.1,
                    0.7,
                    3,
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

        # Always seed from best models (25%)
        print("Seeding population from best previous models (25%)")
        seed_models = model_manager.seed_population_from_best(
            population_size, seed_percentage=0.25
        )

        # Initialize genetic algorithm with enhanced parameters and parallel processing
        # Multi-architecture support with simulated annealing enabled by default
        ga = GeneticAlgorithm(
            population_size=population_size,
            mutation_rate=0.15,
            crossover_rate=0.7,
            elite_size=4,
            games_per_pairing=games_per_pairing,
            num_workers=num_workers,
            parallel_mode=parallel_mode,
            use_cuda=use_cuda,
        )

        print(f"Multi-architecture training enabled: {ga.architecture_enabled}")
        print(f"Simulated annealing temperature: {ga.temperature}")

        # Initialize population
        ga.initialize_population(seed_models)

        # Continuous training loop
        generation = 0
        while not cancellation_flags.get(run_id, False):
            generation += 1
            print(f"\n{'='*60}")
            print(f"Generation {generation} (Continuous Mode)")
            print(f"{'='*60}")

            # Evaluate population
            ga.elo_ratings = ga.evaluate_population_parallel()

            # Print individual model ELO ratings
            for i, elo in enumerate(ga.elo_ratings):
                desc = ga.elo_system.get_rating_description(elo)
                print(f"  Model {i+1}/{len(ga.population)}: ELO = {elo:.0f} ({desc})")

            # Sort by ELO rating
            sorted_pop = sorted(
                zip(ga.population, ga.elo_ratings),
                key=lambda x: x[1],
                reverse=True,
            )

            current_best_model, current_best_elo = sorted_pop[0]
            avg_elo = sum(ga.elo_ratings) / len(ga.elo_ratings)

            # Track improvement for logging
            improved = current_best_elo > ga.global_best_elo

            # Update global champion if current best is better
            if improved:
                ga.global_best_model = copy.deepcopy(current_best_model)
                ga.global_best_elo = current_best_elo
                print(f"\n🏆 NEW GLOBAL CHAMPION! ELO: {current_best_elo:.0f}")

                # Log architecture of champion
                from networks.architecture_registry import ArchitectureRegistry

                arch_type = ArchitectureRegistry.get_architecture_type(
                    current_best_model
                )
                arch_info = ArchitectureRegistry.get_architecture_info(arch_type)
                print(
                    f"   Architecture: {arch_info['name']} ({arch_info['description']})"
                )
            else:
                ga.generations_without_improvement += 1

            # Update temperature and log
            prev_temp = ga.temperature
            ga.update_temperature(improved)
            if abs(ga.temperature - prev_temp) > 0.01:
                temp_change = "↓ cooling" if ga.temperature < prev_temp else "↑ heating"
                print(
                    f"  🌡️  Temperature {temp_change}: {prev_temp:.2f} → {ga.temperature:.2f}"
                )

            # Apply diversity boosts and log when they happen
            if ga.generations_without_improvement == 5:
                ga.mutation_rate = min(ga.base_mutation_rate * 1.5, 0.3)
                ga.apply_diversity_boost(1)
            elif ga.generations_without_improvement == 10:
                ga.mutation_rate = min(ga.base_mutation_rate * 2.0, 0.4)
                ga.apply_diversity_boost(2)
            elif ga.generations_without_improvement >= 15:
                ga.mutation_rate = min(ga.base_mutation_rate * 2.5, 0.5)
                ga.apply_diversity_boost(3)
                ga.generations_without_improvement = 0
            elif ga.generations_without_improvement < 5:
                ga.mutation_rate = ga.base_mutation_rate

            # Update architecture statistics
            ga.update_architecture_stats()

            print(f"\nGeneration {generation} Summary:")
            print(f"  Current Best ELO:  {current_best_elo:.0f}")
            print(f"  Global Best ELO:   {ga.global_best_elo:.0f}")
            print(f"  Average ELO:       {avg_elo:.0f}")
            print(f"  Mutation Rate:     {ga.mutation_rate:.3f}")
            print(f"  Temperature:       {ga.temperature:.2f}")
            print(f"  Gens w/o Improve:  {ga.generations_without_improvement}")

            # Print architecture distribution
            ga.print_architecture_stats()

            # Update training state
            with training_lock:
                training_runs[run_id]["current_generation"] = generation
                training_runs[run_id]["best_fitness"] = ga.global_best_elo
                training_runs[run_id]["avg_fitness"] = avg_elo

            # Update database with generation count and fitness values
            try:
                conn = get_db_connection()
                cur = conn.cursor()
                cur.execute(
                    """
                    UPDATE training_runs 
                    SET generation_count = %s, best_fitness = %s, avg_fitness = %s
                    WHERE id = %s
                """,
                    (generation, ga.global_best_elo, avg_elo, run_id),
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
                        generation,
                        ga.global_best_elo,
                        avg_elo,
                        f"Generation {generation}: Best = {ga.global_best_elo:.0f}, Avg = {avg_elo:.0f}",
                    ),
                )

                conn.commit()
                cur.close()
                conn.close()
            except Exception as e:
                print(f"Error updating training run: {e}")

            # Auto-save best model every 5 generations (and at generation 1)
            if (
                generation == 1 or generation % 5 == 0
            ) and ga.global_best_model is not None:
                print(f"  💾 Auto-saving best model (generation {generation})...")
                model_manager.save_model(
                    ga.global_best_model,
                    f"AutoSave-Gen{generation}",
                    generation,
                    ga.global_best_elo,
                    run_id,
                    is_best=True,
                    elo_rating=ga.global_best_elo,
                )

            # Keep elite - ALWAYS include global champion first
            elites = []
            if ga.global_best_model is not None:
                elites.append(copy.deepcopy(ga.global_best_model))
                print(f"  ✓ Global champion preserved in population")

            # Add remaining elites from current generation
            for model, elo in sorted_pop[: ga.elite_size]:
                if len(elites) < ga.elite_size:
                    elites.append(copy.deepcopy(model))

            # Select parents
            parents = ga.selection()

            # Create offspring
            offspring = []
            for i in range(0, len(parents), 2):
                if i + 1 < len(parents):
                    child1 = ga.crossover(parents[i], parents[i + 1])
                    child2 = ga.crossover(parents[i + 1], parents[i])
                    ga.mutate(child1)
                    ga.mutate(child2)
                    offspring.extend([child1, child2])

            # New population
            ga.population = elites + offspring[: ga.population_size - ga.elite_size]

        # Training was cancelled
        print(f"\n⚠️  Training cancelled at generation {generation}")

        # Save final best model
        if ga.global_best_model is not None:
            print(f"  💾 Saving final best model...")
            model_id = model_manager.save_model(
                ga.global_best_model,
                f"FinalBest-Gen{generation}",
                generation,
                ga.global_best_elo,
                run_id,
                is_best=True,
                elo_rating=ga.global_best_elo,
            )

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
