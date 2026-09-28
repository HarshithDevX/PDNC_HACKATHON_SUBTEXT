import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, classification_report

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from data_loader import load_train_data
from preprocessing import format_text_for_baseline

def run_baseline_experiments():
    print("Loading data for baseline model...")
    df_train = load_train_data()
    
    # Map labels to numeric 0 and 1
    y_all = df_train["label"].map(config.LABEL2ID).values
    
    # Stratified split: 80% train, 20% validation
    sss = StratifiedShuffleSplit(n_splits=1, test_size=config.VAL_SPLIT_SIZE, random_state=config.RANDOM_SEED)
    train_idx, val_idx = next(sss.split(df_train, y_all))
    
    strategies = ["RESPONSE_ONLY", "PARENT_RESPONSE", "FULL_CONTEXT_RESPONSE"]
    results = []

    print(f"\n--- Running TF-IDF + Logistic Regression Baselines ({len(train_idx)} train, {len(val_idx)} val) ---")
    for strat in strategies:
        print(f"\nEvaluating Baseline Strategy: {strat}...")
        texts = [format_text_for_baseline(row, strategy=strat) for _, row in df_train.iterrows()]
        
        train_texts = [texts[i] for i in train_idx]
        val_texts = [texts[i] for i in val_idx]
        
        y_train = y_all[train_idx]
        y_val = y_all[val_idx]
        
        # TF-IDF Vectorization
        vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), sublinear_tf=True)
        X_train = vectorizer.fit_transform(train_texts)
        X_val = vectorizer.transform(val_texts)
        
        # Logistic Regression Classifier
        clf = LogisticRegression(C=1.0, max_iter=1000, random_state=config.RANDOM_SEED)
        clf.fit(X_train, y_train)
        
        # Predictions
        val_preds = clf.predict(X_val)
        
        macro_f1 = f1_score(y_val, val_preds, average="macro")
        sarcasm_f1 = f1_score(y_val, val_preds, pos_label=1)
        acc = accuracy_score(y_val, val_preds)
        
        print(f"Strategy: {strat:22} | Macro F1: {macro_f1:.4f} | Sarcasm F1: {sarcasm_f1:.4f} | Accuracy: {acc:.4f}")
        
        results.append({
            "Model": f"Baseline (TF-IDF + LogReg)",
            "Context_Strategy": strat,
            "Macro_F1": round(macro_f1, 4),
            "Sarcasm_F1": round(sarcasm_f1, 4),
            "Accuracy": round(acc, 4)
        })

    # Save to reports/experiment_results.csv
    df_results = pd.DataFrame(results)
    if os.path.exists(config.EXPERIMENT_RESULTS_FILE):
        df_existing = pd.read_csv(config.EXPERIMENT_RESULTS_FILE)
        # Avoid duplicate baseline rows
        df_combined = pd.concat([df_existing[~df_existing["Model"].str.contains("Baseline")], df_results], ignore_index=True)
    else:
        df_combined = df_results
        
    df_combined.to_csv(config.EXPERIMENT_RESULTS_FILE, index=False)
    print(f"\nExperiment results saved to {config.EXPERIMENT_RESULTS_FILE}")
    return df_results

if __name__ == "__main__":
    run_baseline_experiments()
