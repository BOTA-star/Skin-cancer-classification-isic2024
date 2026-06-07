import os


# ===========
# PATH CONFIG
# ===========

DATA_ROOT = os.getenv("ISIC_DATA_ROOT", "data")

RAW_DIR = os.path.join(DATA_ROOT, "raw")
PROCESSED_DIR = os.path.join(DATA_ROOT, "processed")
SPLIT_DIR = os.path.join(DATA_ROOT, "splits")

# Giữ alias này để các script cũ vẫn chạy được
DATA_DIR = RAW_DIR
WORK_DIR = RAW_DIR

CHECKPOINT_DIR = os.getenv("ISIC_CHECKPOINT_DIR", "checkpoints")
ARTIFACT_DIR = os.getenv("ISIC_ARTIFACT_DIR", "artifacts")
OUTPUT_DIR = os.getenv("ISIC_OUTPUT_DIR", "outputs")


TRAIN_CSV = os.path.join(RAW_DIR, "train-metadata.csv")
TEST_CSV = os.path.join(RAW_DIR, "test-metadata.csv")
SAMPLE_SUBMISSION_CSV = os.path.join(RAW_DIR, "sample_submission.csv")

TRAIN_IMAGE_ZIP = os.path.join(RAW_DIR, "ISIC_2024_Training_Input.zip")
TEST_IMAGE_HDF5 = os.path.join(RAW_DIR, "test-image.hdf5")

TRAIN_IMAGE_DIR = os.path.join(RAW_DIR, "ISIC_2024_Training_Input")


TRAIN_SPLIT_CSV = os.path.join(SPLIT_DIR, "train_split.csv")
VAL_SPLIT_CSV = os.path.join(SPLIT_DIR, "val_split.csv")
HOLDOUT_SPLIT_CSV = os.path.join(SPLIT_DIR, "holdout_split.csv")

TRAIN_PROCESSED_CSV = os.path.join(PROCESSED_DIR, "train_processed.csv")
VAL_PROCESSED_CSV = os.path.join(PROCESSED_DIR, "val_processed.csv")
HOLDOUT_PROCESSED_CSV = os.path.join(PROCESSED_DIR, "holdout_processed.csv")

METADATA_PREPROCESSOR_PATH = os.path.join(
    ARTIFACT_DIR,
    "metadata_preprocessor.pkl"
)

BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best.pth")
LAST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "last.pth")

SUBMISSION_PATH = os.path.join(OUTPUT_DIR, "submission.csv")


# ===========
# TRAINING CONFIG
# ===========

# ===========
# TRAINING CONFIG
# ===========

SEED = 42

# CPU demo mode
IMAGE_SIZE = 160
BATCH_SIZE = 4
EPOCHS = 2
LR = 1e-4

# Chỉ lấy ít data để demo local
NEGATIVE_SAMPLE_SIZE = 500
EVAL_NEGATIVE_SAMPLE_SIZE = 200

VAL_SIZE = 0.2
HOLDOUT_SIZE = 0.1

NUM_WORKERS = 0
PATIENCE = 2

# Giới hạn batch để CPU vẫn chạy nổi
MAX_TRAIN_BATCHES = 20
MAX_VAL_BATCHES = 10

# Local CPU không tải pretrained để tránh nặng
USE_PRETRAINED = False

# Backbone nhẹ cho demo local
BACKBONE = "mobilenet_v3_small"

# Không unfreeze backbone trong demo local
UNFREEZE_EPOCH = 999


# ===========
# METADATA CONFIG
# ===========

NUM_COLS = [
    "age_approx",
    "clin_size_long_diam_mm",

    "tbp_lv_A",
    "tbp_lv_Aext",
    "tbp_lv_B",
    "tbp_lv_Bext",
    "tbp_lv_C",
    "tbp_lv_Cext",
    "tbp_lv_H",
    "tbp_lv_Hext",
    "tbp_lv_L",
    "tbp_lv_Lext",
    "tbp_lv_areaMM2",
    "tbp_lv_area_perim_ratio",
    "tbp_lv_color_std_mean",
    "tbp_lv_deltaA",
    "tbp_lv_deltaB",
    "tbp_lv_deltaL",
    "tbp_lv_deltaLB",
    "tbp_lv_deltaLBnorm",
    "tbp_lv_eccentricity",
    "tbp_lv_minorAxisMM",
    "tbp_lv_nevi_confidence",
    "tbp_lv_norm_border",
    "tbp_lv_norm_color",
    "tbp_lv_perimeterMM",
    "tbp_lv_radial_color_std_max",
    "tbp_lv_stdL",
    "tbp_lv_stdLExt",
    "tbp_lv_symm_2axis",
    "tbp_lv_symm_2axis_angle",
    "tbp_lv_x",
    "tbp_lv_y",
    "tbp_lv_z",
]

CAT_COLS = [
    "sex",
    "anatom_site_general",
    "image_type",
    "tbp_tile_type",
    "tbp_lv_location",
    "tbp_lv_location_simple",
]