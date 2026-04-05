from pathlib import Path
import torch

DATA_DIR       = Path("./data")
OUTPUT_DIR     = Path("./output")

RAW_SAMPLE_PATH   = DATA_DIR / "raw_sample.csv"
USER_PROFILE_PATH = DATA_DIR / "user_profile.csv"
AD_FEATURE_PATH   = DATA_DIR / "ad_feature.csv"
BEHAVIOR_LOG_PATH = DATA_DIR / "behavior_log.csv"

CHUNK_SIZE     = 5_000_000
TRAIN_END_DATE = "2017-05-12"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")