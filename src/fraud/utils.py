import os
import numpy as np
import pandas as pd

def compute_scale_pos_weight(y):
    # Avoid division by zero
    n_pos = max(int((y == 1).sum()), 1)
    n_neg = max(int((y == 0).sum()), 1)
    return n_neg / n_pos

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
