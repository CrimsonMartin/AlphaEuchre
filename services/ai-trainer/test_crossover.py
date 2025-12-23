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

    # Create models of different architectures
    mlp = ArchitectureRegistry.create_model("basic", use_cuda=False)
    cnn = ArchitectureRegistry.create_model("cnn", use_cuda=False)
    transformer = ArchitectureRegistry.create_model("transformer", use_cuda=False)

    print(f"MLP arch: {ArchitectureRegistry.get_architecture_type(mlp)}")
    print(f"CNN arch: {ArchitectureRegistry.get_architecture_type(cnn)}")
    print(
        f"Transformer arch: {ArchitectureRegistry.get_architecture_type(transformer)}"
    )

    # Test crossover between same architecture (MLP + MLP)
    print("\nTesting MLP + MLP crossover...")
    mlp2 = ArchitectureRegistry.create_model("basic", use_cuda=False)
    child_mlp = ga.crossover(mlp, mlp2)
    print(f"Child arch: {ArchitectureRegistry.get_architecture_type(child_mlp)}")
    assert ArchitectureRegistry.get_architecture_type(child_mlp) == "basic"
    print("✓ MLP + MLP crossover successful")

    # Test crossover between different architectures (MLP + CNN)
    print("\nTesting MLP + CNN crossover (should not crash)...")
    child_mixed = ga.crossover(mlp, cnn)
    print(f"Child arch: {ArchitectureRegistry.get_architecture_type(child_mixed)}")
    assert ArchitectureRegistry.get_architecture_type(child_mixed) == "basic"
    print("✓ MLP + CNN crossover successful (handled mismatch)")

    # Test crossover between CNN + Transformer
    print("\nTesting CNN + Transformer crossover (should not crash)...")
    child_mixed2 = ga.crossover(cnn, transformer)
    print(f"Child arch: {ArchitectureRegistry.get_architecture_type(child_mixed2)}")
    assert ArchitectureRegistry.get_architecture_type(child_mixed2) == "cnn"
    print("✓ CNN + Transformer crossover successful (handled mismatch)")

    # Test crossover between CNN + CNN
    print("\nTesting CNN + CNN crossover...")
    cnn2 = ArchitectureRegistry.create_model("cnn", use_cuda=False)
    child_cnn = ga.crossover(cnn, cnn2)
    print(f"Child arch: {ArchitectureRegistry.get_architecture_type(child_cnn)}")
    assert ArchitectureRegistry.get_architecture_type(child_cnn) == "cnn"
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
