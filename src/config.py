from pathlib import Path

# Project Root
ROOT_DIR = Path(__file__).resolve().parent.parent

# Data Paths
RAW_TRAIN_PATH = ROOT_DIR / "data" / "raw" / "train.csv"
RAW_TEST_PATH = ROOT_DIR / "data" / "raw" / "test.csv"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODEL_DIR = ROOT_DIR / "models"

# Reproducibility & Split Settings
RANDOM_STATE = 42
TEST_SIZE = 0.20
TARGET_COL = "SalePrice"
