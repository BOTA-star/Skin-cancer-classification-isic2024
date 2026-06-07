import os
import random

import numpy as np
import torch


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def ensure_dirs(*dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def check_file_exists(path: str, name: str = "file"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {name}: {path}")

def check_dir_exists(path: str, name: str = "directory"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {name}: {path}")