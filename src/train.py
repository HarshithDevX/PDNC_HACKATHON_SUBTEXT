import os
import sys
import json
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.model_selection import StratifiedShuffleSplit

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from data_loader import load_train_data
from dataset import SarcasmDataset
from model import ContextAwareClassifier
from evaluate import compute_metrics, find_best_threshold, generate_error_analysis

def train_model(
    model_name: str = config.MODEL_NAME,
    context_strategy: str = config.DEFAULT_CONTEXT_STRATEGY,
    num_epochs: int = config.NUM_EPOCHS,
    batch_size: int = config.BATCH_SIZE,
    learning_rate: float = config.LEARNING_RATE,
    max_length: int = config.MAX_SEQ_LENGTH
):
    print(f"\n{'='*60}")
    print(f"STARTING TRAINING PIPELINE: {model_name}")
    print(f"Context Strategy: {context_strategy} | Max Length: {max_length}")
    print(f"Epochs: {num_epochs} | Batch Size: {batch_size} | LR: {learning_rate}")
    print(f"{'='*60}")

    # Set seeds for reproducibility
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # Load data
    df_all = load_train_data()
    y_all = df_all["label"].map(config.LABEL2ID).values

    # Stratified validation split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=config.VAL_SPLIT_SIZE, random_state=config.RANDOM_SEED)
    train_idx, val_idx = next(sss.split(df_all, y_all))
    df_train = df_all.iloc[train_idx].reset_index(drop=True)
    df_val = df_all.iloc[val_idx].reset_index(drop=True)
    y_val = y_all[val_idx]

    print(f"Dataset split: {len(df_train)} train examples | {len(df_val)} validation examples")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = ContextAwareClassifier(model_name=model_name, num_labels=2)
    model.to(device)

    # Prepare datasets and loaders
    train_dataset = SarcasmDataset(df_train, tokenizer, strategy=context_strategy, max_length=max_length)
    val_dataset = SarcasmDataset(df_val, tokenizer, strategy=context_strategy, max_length=max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Optimizer & Scheduler
    total_steps = len(train_loader) * num_epochs
    warmup_steps = int(total_steps * config.WARMUP_RATIO)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=config.WEIGHT_DECAY)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    best_macro_f1 = -1.0
    best_epoch = -1
    best_val_probs = None
    epoch_history = []

    for epoch in range(1, num_epochs + 1):
        start_time = time.time()
        model.train()
        total_train_loss = 0.0

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()
            if (step + 1) % 20 == 0 or (step + 1) == len(train_loader):
                print(f"  Step {step + 1}/{len(train_loader)} | Current Loss: {loss.item():.4f}", flush=True)

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation phase
        model.eval()
        total_val_loss = 0.0
        val_probs_list = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                total_val_loss += outputs["loss"].item()
                # Probability of class 1 (SARCASM)
                sarcasm_probs = outputs["probs"][:, 1].cpu().numpy()
                val_probs_list.extend(sarcasm_probs)

        avg_val_loss = total_val_loss / len(val_loader)
        val_probs_arr = np.array(val_probs_list)
        metrics = compute_metrics(y_val, val_probs_arr, threshold=0.50)
        elapsed = time.time() - start_time

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 4),
            "val_loss": round(avg_val_loss, 4),
            "macro_f1": metrics["macro_f1"],
            "sarcasm_f1": metrics["sarcasm_f1"],
            "accuracy": metrics["accuracy"],
            "time_sec": round(elapsed, 1)
        }
        epoch_history.append(epoch_record)

        print(f"Epoch {epoch}/{num_epochs} [{elapsed:.1f}s] | "
              f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
              f"Macro F1: {metrics['macro_f1']:.4f} | Sarcasm F1: {metrics['sarcasm_f1']:.4f} | "
              f"Accuracy: {metrics['accuracy']:.4f}")

        # Primary checkpoint criterion: Validation Macro F1
        if metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = metrics["macro_f1"]
            best_epoch = epoch
            best_val_probs = val_probs_arr
            print(f"  -> New Best Model found at epoch {epoch}! Saving checkpoint...")
            model.save_pretrained(config.BEST_MODEL_DIR)
            tokenizer.save_pretrained(config.BEST_MODEL_DIR)

    print(f"\nTraining Complete! Best epoch: {best_epoch} with Macro F1: {best_macro_f1:.4f}")

    # Phase 9: Threshold Tuning on validation set
    print("\nExecuting Validation Threshold Tuning...")
    thresh_results = find_best_threshold(y_val, best_val_probs)
    selected_threshold = thresh_results["best_threshold"]
    final_val_metrics = thresh_results["best_metrics"]

    print(f"Default (0.50) Val Macro F1: {thresh_results['val_macro_f1_default']:.4f}")
    print(f"Tuned ({selected_threshold:.2f}) Val Macro F1:   {thresh_results['val_macro_f1_tuned']:.4f} "
          f"(Improvement: {thresh_results['improvement']:+.4f})")

    # Generate and save validation predictions CSV
    df_eval = generate_error_analysis(df_val, y_val, best_val_probs, threshold=selected_threshold)
    df_eval.to_csv(config.VAL_PRED_FILE, index=False)
    print(f"Saved validation predictions to {config.VAL_PRED_FILE}")

    # Save model metadata
    metadata = {
        "model_name": model_name,
        "context_strategy": context_strategy,
        "best_epoch": best_epoch,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "max_seq_length": max_length,
        "random_seed": config.RANDOM_SEED,
        "selected_threshold": selected_threshold,
        "metrics": {
            "validation_macro_f1": final_val_metrics["macro_f1"],
            "validation_sarcasm_f1": final_val_metrics["sarcasm_f1"],
            "validation_not_sarcasm_f1": final_val_metrics["not_sarcasm_f1"],
            "validation_accuracy": final_val_metrics["accuracy"],
            "confusion_matrix": final_val_metrics["confusion_matrix"]
        },
        "threshold_tuning": thresh_results,
        "epoch_history": epoch_history
    }

    metadata_path = os.path.join(config.BEST_MODEL_DIR, "model_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    with open(config.METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved model metadata to {metadata_path} and {config.METRICS_FILE}")

    # Append to experiment results table
    exp_entry = {
        "Model": f"Transformer ({model_name})",
        "Context_Strategy": context_strategy,
        "Macro_F1": round(final_val_metrics["macro_f1"], 4),
        "Sarcasm_F1": round(final_val_metrics["sarcasm_f1"], 4),
        "Accuracy": round(final_val_metrics["accuracy"], 4)
    }
    
    if os.path.exists(config.EXPERIMENT_RESULTS_FILE):
        df_exp = pd.read_csv(config.EXPERIMENT_RESULTS_FILE)
        # Avoid duplicate entries for same model and strategy
        df_exp = df_exp[~((df_exp["Model"] == exp_entry["Model"]) & (df_exp["Context_Strategy"] == context_strategy))]
        df_exp = pd.concat([df_exp, pd.DataFrame([exp_entry])], ignore_index=True)
    else:
        df_exp = pd.DataFrame([exp_entry])
    df_exp.to_csv(config.EXPERIMENT_RESULTS_FILE, index=False)
    print(f"Updated {config.EXPERIMENT_RESULTS_FILE}")

    return metadata

if __name__ == "__main__":
    train_model()
