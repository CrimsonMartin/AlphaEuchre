"""
Architecture Registry for Neural Network Models
Factory pattern for creating models of different architectures
"""

import random
from typing import Dict, List, Type
from networks.basic_nn import BasicEuchreNN
from networks.cnn_nn import CNNEuchreNN
from networks.transformer_nn import TransformerEuchreNN


class ArchitectureRegistry:
    """
    Registry for managing different neural network architectures.
    Provides factory methods for creating models and tracking architecture statistics.
    """

    # Registry of available architectures
    # NOTE: Only basic enabled - has shown best results in training
    ARCHITECTURES: Dict[str, Type] = {
        "basic": BasicEuchreNN,  # Enabled - best results
        # "cnn": CNNEuchreNN,           # Disabled
        # "transformer": TransformerEuchreNN,  # Disabled
    }

    # Architecture metadata
    ARCHITECTURE_INFO: Dict[str, Dict] = {
        "basic": {
            "name": "Basic MLP",
            "description": "Multi-layer perceptron with separate heads",
            "complexity": "low",
            "parameter_count": "~50k",
        },
        "cnn": {
            "name": "CNN",
            "description": "Convolutional neural network for pattern detection",
            "complexity": "medium",
            "parameter_count": "~80k",
        },
        "transformer": {
            "name": "Transformer",
            "description": "Self-attention based architecture",
            "complexity": "high",
            "parameter_count": "~120k",
        },
    }

    @classmethod
    def create_model(cls, architecture_type: str | None = None, use_cuda: bool = True):
        """
        Create a model of the specified architecture type.

        Args:
            architecture_type: Type of architecture ('basic', 'cnn', 'transformer')
                             If None, randomly selects an architecture
            use_cuda: Whether to use CUDA if available

        Returns:
            Neural network model instance

        Raises:
            ValueError: If architecture type is not recognized
        """
        if architecture_type is None:
            architecture_type = random.choice(list(cls.ARCHITECTURES.keys()))
        elif architecture_type not in cls.ARCHITECTURES:
            architecture_type = random.choice(list(cls.ARCHITECTURES.keys()))

        model_class = cls.ARCHITECTURES[architecture_type]
        return model_class(use_cuda=use_cuda)

    @classmethod
    def create_random_model(cls, use_cuda: bool = True):
        """
        Create a model with a randomly selected architecture.

        Args:
            use_cuda: Whether to use CUDA if available

        Returns:
            Neural network model instance
        """
        architecture_type = random.choice(list(cls.ARCHITECTURES.keys()))
        return cls.create_model(architecture_type, use_cuda)

    @classmethod
    def create_population(
        cls,
        size: int,
        architecture_distribution: Dict[str, float] = None,
        use_cuda: bool = True,
    ) -> List:
        """
        Create a population of models with specified architecture distribution.

        Args:
            size: Number of models to create
            architecture_distribution: Dict mapping architecture type to proportion
                                      e.g., {'basic': 0.5, 'cnn': 0.3, 'transformer': 0.2}
                                      If None, uses equal distribution
            use_cuda: Whether to use CUDA if available

        Returns:
            List of neural network models
        """
        if architecture_distribution is None:
            # Equal distribution
            num_architectures = len(cls.ARCHITECTURES)
            architecture_distribution = {
                arch: 1.0 / num_architectures for arch in cls.ARCHITECTURES.keys()
            }

        # Normalize distribution to sum to 1.0
        total = sum(architecture_distribution.values())
        architecture_distribution = {
            k: v / total for k, v in architecture_distribution.items()
        }

        # Create population
        population = []
        remaining = size

        # Assign models to each architecture
        for arch_type, proportion in architecture_distribution.items():
            count = int(size * proportion)
            for _ in range(count):
                population.append(cls.create_model(arch_type, use_cuda))
            remaining -= count

        # Fill remaining slots with random architectures
        for _ in range(remaining):
            population.append(cls.create_random_model(use_cuda))

        # Shuffle to mix architectures
        random.shuffle(population)

        return population

    @classmethod
    def get_architecture_type(cls, model) -> str:
        """
        Get the architecture type of a model.

        Args:
            model: Neural network model instance

        Returns:
            Architecture type string
        """
        if hasattr(model, "architecture_type"):
            return model.architecture_type

        # Fallback: check class name
        class_name = model.__class__.__name__
        if "CNN" in class_name:
            return "cnn"
        elif "Transformer" in class_name:
            return "transformer"
        else:
            return "basic"

    @classmethod
    def get_architecture_info(cls, architecture_type: str) -> Dict:
        """
        Get metadata about an architecture type.

        Args:
            architecture_type: Type of architecture

        Returns:
            Dictionary with architecture information
        """
        return cls.ARCHITECTURE_INFO.get(
            architecture_type,
            {"name": "Unknown", "description": "Unknown architecture"},
        )

    @classmethod
    def get_available_architectures(cls) -> List[str]:
        """
        Get list of available architecture types.

        Returns:
            List of architecture type strings
        """
        return list(cls.ARCHITECTURES.keys())

    @classmethod
    def count_architectures(cls, population: List) -> Dict[str, int]:
        """
        Count the number of models of each architecture type in a population.

        Args:
            population: List of neural network models

        Returns:
            Dictionary mapping architecture type to count
        """
        counts = {arch: 0 for arch in cls.ARCHITECTURES.keys()}

        for model in population:
            arch_type = cls.get_architecture_type(model)
            if arch_type in counts:
                counts[arch_type] += 1
            else:
                counts["unknown"] = counts.get("unknown", 0) + 1

        return counts

    @classmethod
    def get_population_diversity(cls, population: List) -> float:
        """
        Calculate architecture diversity in a population (0.0 to 1.0).

        Args:
            population: List of neural network models

        Returns:
            Diversity score (higher = more diverse)
        """
        if not population:
            return 0.0

        counts = cls.count_architectures(population)
        total = len(population)

        # Calculate Shannon entropy as diversity measure
        diversity = 0.0
        for count in counts.values():
            if count > 0:
                proportion = count / total
                diversity -= proportion * (proportion**0.5)  # Simplified entropy

        # Normalize to 0-1 range
        max_diversity = len(cls.ARCHITECTURES) ** 0.5
        return min(diversity / max_diversity, 1.0) if max_diversity > 0 else 0.0
