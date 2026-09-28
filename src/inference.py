import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from data_loader import load_test_data
from dataset import SarcasmDataset
from model import ContextAwareClassifier
from validator import validate_submission

def run_test_inference(model_dir: str = config.BEST_MODEL_DIR):
    print(f"\n{'='*60}")
    print("RUNNING TEST INFERENCE ON OFFICIAL 2,000 TEST EXAMPLES")
    print(f"Loading checkpoint from: {model_dir}")
    print(f"{'='*60}")

    metadata_path = os.path.join(model_dir, "model_metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        threshold = metadata.get("selected_threshold", 0.50)
        strategy = metadata.get("context_strategy", config.DEFAULT_CONTEXT_STRATEGY)
        max_length = metadata.get("max_seq_length", config.MAX_SEQ_LENGTH)
        print(f"Using tuned threshold: {threshold:.4f} | Strategy: {strategy}")
    else:
        threshold = 0.50
        strategy = config.DEFAULT_CONTEXT_STRATEGY
        max_length = config.MAX_SEQ_LENGTH
        print(f"Metadata not found; falling back to default threshold {threshold} and strategy {strategy}")

    # Load test data
    df_test = load_test_data()
    print(f"Loaded {len(df_test):,} test examples from {config.TEST_FILE}")

    # Load tokenizer and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = ContextAwareClassifier.from_pretrained(model_dir, num_labels=2)
    model.to(device)
    model.eval()

    # Create dataset & loader
    test_dataset = SarcasmDataset(
        df_test,
        tokenizer,
        strategy=strategy,
        max_length=max_length,
        is_test=True
    )
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    all_probs = []
    all_ids = []

    print("Generating predictions...")
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            ids = list(batch["id"])

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            # Sarcasm probability (class 1)
            sarcasm_probs = outputs["probs"][:, 1].cpu().numpy()

            all_probs.extend(sarcasm_probs)
            all_ids.extend(ids)

    all_probs = np.array(all_probs)
    # Apply tuned decision threshold
    pred_labels = (all_probs >= threshold).astype(int)
    pred_strings = [config.ID2LABEL[p] for p in pred_labels]
    confidences = np.where(pred_labels == 1, all_probs, 1.0 - all_probs)

    # 1. Rich test prediction file for debugging/explainability
    df_detailed = df_test.copy()
    df_detailed["prediction"] = pred_strings
    df_detailed["sarcasm_prob"] = np.round(all_probs, 4)
    df_detailed["confidence"] = np.round(confidences, 4)
    df_detailed.to_csv(config.TEST_PRED_FILE, index=False)
    print(f"Saved detailed predictions to {config.TEST_PRED_FILE}")

    # 2. Strict Official Hackathon Submission: ONLY id and prediction
    df_submission = pd.DataFrame({
        "id": all_ids,
        "prediction": pred_strings
    })
    df_submission.to_csv(config.SUBMISSION_FILE, index=False)
    print(f"Saved official submission to {config.SUBMISSION_FILE}")

    # 3. Automatically run official validator
    print("\nRunning Validator against generated submission...")
    is_valid, errors = validate_submission(config.SUBMISSION_FILE)
    if not is_valid:
        raise ValueError(f"Submission validation failed: {errors}")

    return config.SUBMISSION_FILE

if __name__ == "__main__":
    run_test_inference()
