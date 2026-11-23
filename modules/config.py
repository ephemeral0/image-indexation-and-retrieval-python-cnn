
from pathlib import Path

# Project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Dataset directories
DATASETS_DIR = BASE_DIR / "datasets"

DATASET_PATHS = {
    "Corel-1K": DATASETS_DIR / "Corel-1K",
    "Corel-5K": DATASETS_DIR / "Corel-5K",
    "Corel-10K": DATASETS_DIR / "Corel-10K",
    "CIFAR-100": DATASETS_DIR / "Cifar",
    "Fashion-MNIST": DATASETS_DIR / "Fashion-MNIST"
}

# Feature storage
FEATURES_DIR = BASE_DIR / "features"
FEATURES_DIR.mkdir(parents=True, exist_ok=True)

# Image settings
IMAGE_SIZE = (224, 224)

# Retrieval settings
TOP_K = 10
