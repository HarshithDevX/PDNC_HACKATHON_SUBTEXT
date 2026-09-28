import os
import sys
import pytest
import pandas as pd
import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import config
from data_loader import load_train_data, load_test_data
from preprocessing import build_context_pair, clean_text, truncate_dialogue_history
from dataset import SarcasmDataset
from model import ContextAwareClassifier
from validator import validate_submission
from evaluate import compute_metrics, find_best_threshold

def test_01_data_loading():
    """Verify official train and test dataset files exist and have correct schemas."""
    df_train = load_train_data()
    df_test = load_test_data()

    assert len(df_train) == 4200, f"Expected 4,200 train rows, got {len(df_train)}"
    assert len(df_test) == 2000, f"Expected 2,000 test rows, got {len(df_test)}"
    
    assert "parent_context" in df_train.columns
    assert "full_dialogue_context" in df_train.columns
    assert "response" in df_train.columns
    assert "label" in df_train.columns
    
    # Check label balance
    counts = df_train["label"].value_counts()
    assert counts["SARCASM"] == 2100
    assert counts["NOT_SARCASM"] == 2100

def test_02_context_construction():
    """Verify configurable context construction strategies."""
    sample = {
        "parent_context": "Immediate parent turn.",
        "full_dialogue_context": "Turn 1 [SEP] Turn 2 [SEP] Immediate parent turn.",
        "response": "Target sarcastic comment."
    }

    # Strategy 1: Response only
    ctx, resp = build_context_pair(sample, strategy="RESPONSE_ONLY")
    assert ctx == ""
    assert resp == "Target sarcastic comment."

    # Strategy 2: Parent + Response
    ctx, resp = build_context_pair(sample, strategy="PARENT_RESPONSE")
    assert ctx == "Immediate parent turn."

    # Strategy 3: Full dialogue + Response
    ctx, resp = build_context_pair(sample, strategy="FULL_CONTEXT_RESPONSE")
    assert "Turn 1" in ctx
    assert resp == "Target sarcastic comment."

def test_03_truncation_preserves_target():
    """Verify long dialogue truncation drops older turns and preserves recent parent."""
    long_dialogue = "Turn 1 [SEP] Turn 2 [SEP] Turn 3 [SEP] Turn 4 [SEP] Turn 5 [SEP] Turn 6 [SEP] Turn 7"
    truncated = truncate_dialogue_history(long_dialogue, max_turns=3)
    assert "Turn 7" in truncated
    assert "Turn 6" in truncated
    assert "Turn 5" in truncated
    assert "Turn 1" not in truncated

def test_04_label_mapping():
    """Verify label-to-id and id-to-label consistency."""
    assert config.LABEL2ID["NOT_SARCASM"] == 0
    assert config.LABEL2ID["SARCASM"] == 1
    assert config.ID2LABEL[0] == "NOT_SARCASM"
    assert config.ID2LABEL[1] == "SARCASM"

def test_05_metrics_computation():
    """Verify Macro F1, Sarcasm F1, and Accuracy calculations."""
    y_true = np.array([1, 0, 1, 0, 1])
    y_probs = np.array([0.9, 0.1, 0.8, 0.4, 0.2])
    # with threshold 0.5: preds = [1, 0, 1, 0, 0] -> 4 correct out of 5
    m = compute_metrics(y_true, y_probs, threshold=0.5)
    assert m["accuracy"] == 0.8
    assert m["macro_f1"] > 0.7
    assert "TP" in m["confusion_matrix"]

def test_06_threshold_tuning():
    """Verify threshold tuning optimizes Macro F1 on validation data."""
    y_true = np.array([1, 1, 0, 0, 1, 0])
    y_probs = np.array([0.65, 0.55, 0.45, 0.35, 0.75, 0.25])
    tuning = find_best_threshold(y_true, y_probs)
    assert "best_threshold" in tuning
    assert "val_macro_f1_tuned" in tuning

def test_07_validator_logic():
    """Verify official organizer validator accepts valid schema and rejects invalid schemas."""
    # Test valid dummy dataframe
    dummy_valid = pd.DataFrame({
        "id": range(1, 2001),
        "prediction": ["SARCASM" if i % 2 == 0 else "NOT_SARCASM" for i in range(1, 2001)]
    })
    temp_valid_path = os.path.join(config.OUTPUTS_DIR, "temp_test_valid.csv")
    dummy_valid.to_csv(temp_valid_path, index=False)

    is_valid, errors = validate_submission(temp_valid_path)
    assert is_valid is True
    assert len(errors) == 0

    # Test invalid row count
    dummy_invalid = pd.DataFrame({"id": [1, 2], "prediction": ["SARCASM", "NOT_SARCASM"]})
    temp_invalid_path = os.path.join(config.OUTPUTS_DIR, "temp_test_invalid.csv")
    dummy_invalid.to_csv(temp_invalid_path, index=False)

    is_valid_inv, errors_inv = validate_submission(temp_invalid_path)
    assert is_valid_inv is False
    assert any("row count" in e.lower() for e in errors_inv)

    # Clean up temp files
    if os.path.exists(temp_valid_path): os.remove(temp_valid_path)
    if os.path.exists(temp_invalid_path): os.remove(temp_invalid_path)

if __name__ == "__main__":
    pytest.main(["-v", __file__])
