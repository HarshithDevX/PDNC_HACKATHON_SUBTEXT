import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config

def compute_metrics(y_true, y_probs, threshold: float = 0.5):
    """
    Computes primary (Macro F1) and secondary metrics given probabilities and threshold.
    """
    y_preds = (np.array(y_probs) >= threshold).astype(int)
    
    macro_f1 = f1_score(y_true, y_preds, average="macro", zero_division=0)
    sarcasm_f1 = f1_score(y_true, y_preds, pos_label=1, zero_division=0)
    not_sarcasm_f1 = f1_score(y_true, y_preds, pos_label=0, zero_division=0)
    accuracy = accuracy_score(y_true, y_preds)
    
    cm = confusion_matrix(y_true, y_preds, labels=[0, 1])
    # cm: [[TN, FP], [FN, TP]]
    tn, fp, fn, tp = cm.ravel()
    
    return {
        "threshold": round(float(threshold), 4),
        "macro_f1": round(float(macro_f1), 4),
        "sarcasm_f1": round(float(sarcasm_f1), 4),
        "not_sarcasm_f1": round(float(not_sarcasm_f1), 4),
        "accuracy": round(float(accuracy), 4),
        "confusion_matrix": {
            "TP": int(tp), "FP": int(fp),
            "TN": int(tn), "FN": int(fn)
        }
    }

def find_best_threshold(y_true, y_probs):
    """
    Scans candidate probability thresholds on validation set to maximize validation Macro F1.
    """
    threshold_candidates = np.arange(0.20, 0.85, 0.02)
    best_thresh = 0.50
    best_macro = -1.0
    tuning_log = []
    
    for th in threshold_candidates:
        m = compute_metrics(y_true, y_probs, threshold=th)
        tuning_log.append(m)
        if m["macro_f1"] > best_macro:
            best_macro = m["macro_f1"]
            best_thresh = th

    default_metrics = compute_metrics(y_true, y_probs, threshold=0.50)
    best_metrics = compute_metrics(y_true, y_probs, threshold=best_thresh)
    
    return {
        "default_threshold": 0.50,
        "best_threshold": round(float(best_thresh), 4),
        "val_macro_f1_default": default_metrics["macro_f1"],
        "val_macro_f1_tuned": best_metrics["macro_f1"],
        "improvement": round(best_metrics["macro_f1"] - default_metrics["macro_f1"], 4),
        "best_metrics": best_metrics
    }

def generate_error_analysis(df_val, y_true, y_probs, threshold: float = 0.5):
    """
    Analyzes False Positives, False Negatives, and Hard Cases.
    Saves results to reports/error_analysis.md.
    """
    y_preds = (np.array(y_probs) >= threshold).astype(int)
    confidences = np.where(y_preds == 1, y_probs, 1.0 - np.array(y_probs))
    
    df_eval = df_val.copy().reset_index(drop=True)
    df_eval["y_true"] = y_true
    df_eval["y_pred"] = y_preds
    df_eval["sarcasm_prob"] = np.round(y_probs, 4)
    df_eval["confidence"] = np.round(confidences, 4)
    df_eval["pred_label"] = [config.ID2LABEL[p] for p in y_preds]
    df_eval["actual_label"] = [config.ID2LABEL[y] for y in y_true]
    df_eval["correct"] = (df_eval["y_true"] == df_eval["y_pred"])
    
    fps = df_eval[(df_eval["y_true"] == 0) & (df_eval["y_pred"] == 1)]
    fns = df_eval[(df_eval["y_true"] == 1) & (df_eval["y_pred"] == 0)]
    tps = df_eval[(df_eval["y_true"] == 1) & (df_eval["y_pred"] == 1)]
    tns = df_eval[(df_eval["y_true"] == 0) & (df_eval["y_pred"] == 0)]
    
    # Hard cases: lowest confidence or incorrect with high margin
    hard_cases = df_eval.sort_values(by="confidence", ascending=True).head(10)
    
    # Build Markdown report
    lines = [
        "# SUBTEXT — Model Error Analysis & Technical Defense Report",
        "**Challenge 03:** Context-Aware Sarcasm & Irony Detection",
        f"**Threshold Used:** {threshold:.2f}",
        "",
        "---",
        "## 1. Confusion Matrix Breakdown",
        f"- **True Positives (TP - Correct Sarcasm):** {len(tps):,}",
        f"- **True Negatives (TN - Correct Sincere):** {len(tns):,}",
        f"- **False Positives (FP - Sincere misclassified as Sarcasm):** {len(fps):,}",
        f"- **False Negatives (FN - Sarcasm misclassified as Sincere):** {len(fns):,}",
        "",
        "---",
        "## 2. Hard-Case Analysis (Lowest Confidence / Pragmatic Ambiguity)",
        "The examples below exhibit high pragmatic ambiguity, subtle ironic incongruity, or deceptive sentiment lexical cues:",
        ""
    ]
    
    for i, (_, row) in enumerate(hard_cases.iterrows()):
        reason = "Subtle tone without explicit contrast markers"
        if "?" in row["response"] or "!" in row["response"]:
            reason = "Punctuation-driven ambiguity; surface sarcasm vs genuine enthusiasm"
        elif len(row["parent_context"].split()) < 4:
            reason = "Sparse parent context provides insufficient pragmatic framing"
        elif row["confidence"] < 0.55:
            reason = "Boundary case near the 0.50 decision threshold"

        lines.extend([
            f"### Case #{i+1} (ID: {row.get('id', 'N/A')})",
            f"- **Parent Context:** *\"{row['parent_context']}\"*",
            f"- **Target Response:** *\"{row['response']}\"*",
            f"- **Ground Truth:** `{row['actual_label']}` | **Model Prediction:** `{row['pred_label']}`",
            f"- **Prediction Confidence:** `{row['confidence']*100:.1f}%` (Sarcasm Prob: `{row['sarcasm_prob']:.3f}`)",
            f"- **Context Evidence / Diagnosis:** {reason}",
            ""
        ])

    lines.extend([
        "---",
        "## 3. Key Error Patterns & Mitigation for 2-Minute Defense",
        "1. **Shallow Lexical Cues vs. Pragmatic Incongruity:** Bag-of-words baselines trigger false positives on words like *'obviously'*, *'definitely'*, or *'great'*. The transformer encoder cross-attends the parent context to distinguish literal affirmations from sarcastic incongruity.",
        "2. **Sparse Context Vulnerability:** When parent context is one or two words, contextual incongruity signals are attenuated, leading to lower model confidence.",
        "3. **Tone Polarization:** Invertive sarcasm where positive words express negative evaluation remains harder than hyperbolic exaggeration.",
        ""
    ])

    report_text = "\n".join(lines)
    with open(config.ERROR_ANALYSIS_FILE, "w", encoding="utf-8") as f:
        f.write(report_text)
        
    print(f"Error analysis report successfully saved to {config.ERROR_ANALYSIS_FILE}")
    return df_eval
