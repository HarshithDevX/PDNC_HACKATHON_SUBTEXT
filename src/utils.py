import os
import random
import numpy as np
import torch

def set_seed(seed: int = 42):
    """Ensure reproducibility across random, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def format_percentage(val: float) -> str:
    """Format float 0.0-1.0 to percentage string."""
    return f"{val * 100:.2f}%"

def get_device():
    """Detect available compute device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
