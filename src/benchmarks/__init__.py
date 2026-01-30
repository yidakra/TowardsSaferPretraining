"""Benchmark evaluators + metrics (Table computations)."""

from .metrics import calculate_metrics, print_metrics

# Optional imports: some benchmarks depend on extra packages (e.g., OpenAI SDK).
# Keep `src.benchmarks.metrics` importable in minimal environments.
try:  # pragma: no cover
    from .havoc import HAVOCBenchmark, HAVOCEvaluationResult, LeakageResult
except Exception:  # pragma: no cover
    HAVOCBenchmark = None
    HAVOCEvaluationResult = None
    LeakageResult = None

try:  # pragma: no cover
    from .generators import TransformersGenerator, create_generator
except Exception:  # pragma: no cover
    TransformersGenerator = None
    create_generator = None

__all__ = [
    "calculate_metrics",
    "print_metrics",
    "HAVOCBenchmark",
    "HAVOCEvaluationResult",
    "LeakageResult",
    "TransformersGenerator",
    "create_generator",
]

