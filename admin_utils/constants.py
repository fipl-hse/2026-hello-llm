"""
Useful constant variables.
"""

import os
from decimal import Decimal
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", str(os.cpu_count()))
os.environ.setdefault("MKL_NUM_THREADS", str(os.cpu_count()))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

# pylint: disable=invalid-name,too-few-public-methods
try:
    import torch
except ImportError:
    print('Library "torch" not installed. Failed to import.')

    class torch:
        """
        Mock class for public repo that does not have requirements.
        """

        class cuda:
            """
            Mock class for torch.cuda.
            """

            @staticmethod
            def is_available() -> bool:
                """
                Mock method for torch.cuda.is_available.
                """
                return False


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Selected {DEVICE} for all reference collection tasks")

if DEVICE == "cuda":
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(os.cpu_count() or 1)

PROJECT_ROOT = Path(__file__).parent.parent
PROJECT_CONFIG_PATH = PROJECT_ROOT / "project_config.json"
CONFIG_PACKAGE_PATH = PROJECT_ROOT / "config"

DIST_PATH = PROJECT_ROOT / "dist"

GLOBAL_SEED = 77
GLOBAL_NUM_SAMPLES = 100
GLOBAL_MAX_LENGTH = 120
GLOBAL_INFERENCE_BATCH_SIZE = 64
GLOBAL_BATCH_SIZE_TEST = 3
GLOBAL_FINE_TUNING_BATCH_SIZE = 3
GLOBAL_RANK = 8
GLOBAL_ALPHA = 8
GLOBAL_FINE_TUNING_STEPS = 50
GLOBAL_LEARNING_RATE = 1e-3
GLOBAL_TARGET_MODULES = None
GLOBAL_LORA_DROPOUT = 0.0

MAX_CONTEXT_LENGTH_DEFAULT = 20

QUANTIZATION_EXP = Decimal("1.00000")
