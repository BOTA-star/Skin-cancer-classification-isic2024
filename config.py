from pathlib import Path

# Root directory (project root)
BASE_DIR = Path(__file__).resolve().parent

# =========================
# DATA DIRECTORIES
# =========================
DATA_DIR = BASE_DIR / "data"

RAW_DATA_DIR = DATA_DIR / "raw"
SPLIT_DATA_DIR = DATA_DIR / "splits"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# =========================
# IMAGE PATHS
# =========================
IMAGE_DIR = RAW_DATA_DIR / "ISIC_2024_Training_Input"

# =========================
# FILE PATHS
# =========================
TRAIN_METADATA = RAW_DATA_DIR / "train-metadata.csv"
TEST_METADATA = RAW_DATA_DIR / "test-metadata.csv"

SAMPLE_SUBMISSION = RAW_DATA_DIR / "sample_submission.csv"

# =========================
# SPLIT FILES
# =========================
TRAIN_SPLIT = SPLIT_DATA_DIR / "train_split.csv"
VAL_SPLIT = SPLIT_DATA_DIR / "val_split.csv"
TEST_SPLIT = SPLIT_DATA_DIR / "test_split.csv"

# =========================
# PROCESSED FILES
# =========================
TRAIN_PROCESSED = PROCESSED_DATA_DIR / "train_processed.csv"
VAL_PROCESSED = PROCESSED_DATA_DIR / "val_processed.csv"
TEST_PROCESSED = PROCESSED_DATA_DIR / "test_processed.csv"

# =========================
# RANDOM SEED
# =========================
RANDOM_STATE = 42