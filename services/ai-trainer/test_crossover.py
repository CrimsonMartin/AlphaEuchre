import sys
import os
import torch
import random

# Add paths for imports
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "shared"))

from genetic.genetic_algorithm import GeneticAlgorithm
from networks.architecture_registry import ArchitectureRegistry


def test_crossover():
    print("Testing crossover with different architectures...")
    ga = GeneticAlgorithm(population_size=4, use_cuda=False)

    available = ArchitectureRegistry.get_available_architectures()
    print(f"Available architectures: {available}")

    # Create models - note: if cnn/transformer are disabled in registry,
    # create_model falls back to "basic"
    mlp = ArchitectureRegistry.create_model("basic", use_cuda=False)
    cnn = ArchitectureRegistry.create_model("cnn", use_cuda=False)
    transformer = ArchitectureRegistry.create_model("transformer", use_cuda=False)

    mlp_type = ArchitectureRegistry.get_architecture_type(mlp)
    cnn_type = ArchitectureRegistry.get_architecture_type(cnn)
    transformer_type = ArchitectureRegistry.get_architecture_type(transformer)

    print(f"MLP arch: {mlp_type}")
    print(f"CNN arch: {cnn_type}")
    print(f"Transformer arch: {transformer_type}")

    # Test crossover between same architecture (MLP + MLP)
    print("\nTesting MLP + MLP crossover...")
    mlp2 = ArchitectureRegistry.create_model("basic", use_cuda=False)
    child_mlp = ga.crossover(mlp, mlp2)
    child_mlp_type = ArchitectureRegistry.get_architecture_type(child_mlp)
    print(f"Child arch: {child_mlp_type}")
    assert child_mlp_type == "basic"
    print("✓ MLP + MLP crossover successful")

    # Test crossover between different architectures (MLP + CNN)
    print("\nTesting MLP + CNN crossover (should not crash)...")
    child_mixed = ga.crossover(mlp, cnn)
    child_mixed_type = ArchitectureRegistry.get_architecture_type(child_mixed)
    print(f"Child arch: {child_mixed_type}")
    # When both are actually basic (cnn disabled), child is basic
    # When cnn is enabled, cross-architecture crossover uses parent1's type (basic)
    assert child_mixed_type == "basic"
    print("✓ MLP + CNN crossover successful (handled mismatch)")

    # Test crossover between CNN + Transformer
    print("\nTesting CNN + Transformer crossover (should not crash)...")
    child_mixed2 = ga.crossover(cnn, transformer)
    child_mixed2_type = ArchitectureRegistry.get_architecture_type(child_mixed2)
    print(f"Child arch: {child_mixed2_type}")
    # Child inherits parent1's architecture type
    assert child_mixed2_type == cnn_type
    print("✓ CNN + Transformer crossover successful (handled mismatch)")

    # Test crossover between CNN + CNN
    print("\nTesting CNN + CNN crossover...")
    cnn2 = ArchitectureRegistry.create_model("cnn", use_cuda=False)
    child_cnn = ga.crossover(cnn, cnn2)
    child_cnn_type = ArchitectureRegistry.get_architecture_type(child_cnn)
    print(f"Child arch: {child_cnn_type}")
    assert child_cnn_type == cnn_type
    print("✓ CNN + CNN crossover successful")


if __name__ == "__main__":
    try:
        test_crossover()
        print("\n✨ All crossover tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
