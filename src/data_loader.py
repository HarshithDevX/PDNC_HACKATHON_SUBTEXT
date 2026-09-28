import pandas as pd
import numpy as np
import os
import sys

# Ensure local imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config

def load_train_data(path=None):
    """Load training data from CSV and validate expected columns."""
    filepath = path or config.TRAIN_FILE
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Training dataset not found at {filepath}")
    df = pd.read_csv(filepath)
    # Fill NAs in text columns with empty strings
    for col in ["parent_context", "full_dialogue_context", "response"]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)
    return df

def load_test_data(path=None):
    """Load test data from CSV and validate expected columns."""
    filepath = path or config.TEST_FILE
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Test dataset not found at {filepath}")
    df = pd.read_csv(filepath)
    for col in ["parent_context", "full_dialogue_context", "response"]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)
    return df

def inspect_dataset(df, name="Dataset"):
    """Print comprehensive summary of dataset statistics."""
    stats = {
        "name": name,
        "num_rows": len(df),
        "num_cols": len(df.columns),
        "columns": list(df.columns),
        "missing_values": df.isnull().sum().to_dict(),
        "duplicate_rows": df.duplicated().sum(),
        "duplicate_responses": df["response"].duplicated().sum() if "response" in df.columns else 0,
    }
    if "label" in df.columns:
        stats["label_distribution"] = df["label"].value_counts().to_dict()
    return stats

if __name__ == "__main__":
    df_train = load_train_data()
    df_test = load_test_data()
    print("--- Train Inspection ---")
    print(inspect_dataset(df_train, "Train"))
    print("\n--- Test Inspection ---")
    print(inspect_dataset(df_test, "Test"))
