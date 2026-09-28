import os
import sys
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config

def get_hard_cases(val_pred_file: str = config.VAL_PRED_FILE, top_n: int = 8):
    """
    Extracts the most informative hard cases from the validation predictions:
      - Low confidence predictions
      - Small margin boundaries
      - Pragmatic incongruity edge cases
    """
    if not os.path.exists(val_pred_file):
        return []

    df = pd.read_csv(val_pred_file)
    # Ensure confidence column exists
    if "confidence" not in df.columns:
        df["confidence"] = np.where(df["y_pred"] == 1, df["sarcasm_prob"], 1.0 - df["sarcasm_prob"])

    # Hard cases: lowest confidence
    hard_df = df.sort_values(by="confidence", ascending=True).head(top_n)
    
    cases = []
    for _, row in hard_df.iterrows():
        # Pragmatic reason diagnosis
        reason = "Boundary case near the classification threshold with weak contextual incongruity signal."
        if "?" in str(row["response"]) or "!" in str(row["response"]):
            reason = "Expressive punctuation creates conflicting sentiment signals versus parent context."
        elif len(str(row["parent_context"]).split()) < 4:
            reason = "Sparse parent context provides insufficient conversational grounding."
        elif row["y_true"] != row["y_pred"]:
            reason = "Pragmatic reversal: surface lexical sentiment opposes the conversational intent."

        cases.append({
            "id": int(row.get("id", 0)),
            "parent_context": str(row.get("parent_context", "")),
            "full_dialogue_context": str(row.get("full_dialogue_context", "")),
            "response": str(row.get("response", "")),
            "ground_truth": str(row.get("actual_label", "")),
            "prediction": str(row.get("pred_label", "")),
            "sarcasm_prob": float(row.get("sarcasm_prob", 0.5)),
            "confidence": float(row.get("confidence", 0.5)),
            "confidence_percent": f"{float(row.get('confidence', 0.5)) * 100:.1f}%",
            "is_correct": bool(row.get("correct", True)),
            "why_hard_case": reason
        })
    return cases

def extract_context_evidence(parent_context: str, full_dialogue_context: str, response: str):
    """
    Formats the context evidence layers used by the model.
    Highlights turn structure and immediate conversational dependencies.
    """
    turns = []
    if full_dialogue_context:
        raw_turns = full_dialogue_context.split(" [SEP] ")
        for i, t in enumerate(raw_turns):
            turns.append({
                "turn_index": i + 1,
                "text": t.strip(),
                "is_immediate_parent": (t.strip() == parent_context.strip())
            })
    elif parent_context:
        turns.append({
            "turn_index": 1,
            "text": parent_context.strip(),
            "is_immediate_parent": True
        })

    return {
        "dialogue_turns": turns,
        "immediate_parent": parent_context,
        "target_response": response,
        "input_representation": f"[CLS] {parent_context} [SEP] {response} [SEP]"
    }
