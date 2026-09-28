import os
import sys
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from model import ContextAwareClassifier
from preprocessing import build_context_pair

def run_single_ablation(row, model, tokenizer, device, threshold: float = 0.5):
    """
    Runs the same target response through:
      1. RESPONSE_ONLY
      2. PARENT_RESPONSE
      3. FULL_CONTEXT_RESPONSE
    Returns predictions, probabilities, and confidence for each strategy.
    """
    strategies = ["RESPONSE_ONLY", "PARENT_RESPONSE", "FULL_CONTEXT_RESPONSE"]
    results = {}

    model.eval()
    with torch.no_grad():
        for strat in strategies:
            context, response = build_context_pair(row, strategy=strat)
            if not context:
                encoding = tokenizer(
                    text=response,
                    max_length=config.MAX_SEQ_LENGTH,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                )
            else:
                encoding = tokenizer(
                    text=context,
                    text_pair=response,
                    max_length=config.MAX_SEQ_LENGTH,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                )
            
            input_ids = encoding["input_ids"].to(device)
            attention_mask = encoding["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            sarcasm_prob = float(outputs["probs"][0, 1].item())
            
            pred_id = int(sarcasm_prob >= threshold)
            pred_label = config.ID2LABEL[pred_id]
            confidence = sarcasm_prob if pred_id == 1 else (1.0 - sarcasm_prob)

            results[strat] = {
                "strategy": strat,
                "context_used": context if context else "(None - Response Isolated)",
                "response": response,
                "prediction": pred_label,
                "sarcasm_probability": round(sarcasm_prob, 4),
                "confidence": round(confidence, 4),
                "confidence_percent": f"{confidence * 100:.1f}%"
            }

    return results

def get_curated_ablation_examples():
    """
    Returns high-impact conversational examples where context resolves ambiguity.
    """
    return [
        {
            "id": "ABL-01",
            "full_dialogue_context": "The software engineer pushed untested code directly into production on a Friday at 5 PM. [SEP] The entire payment system crashed for 4 hours.",
            "parent_context": "The entire payment system crashed for 4 hours.",
            "response": "Oh brilliant job, that was truly a masterpiece of engineering!",
            "note": "Isolated response appears enthusiastic/praise; parent context reveals severe failure making praise ironic."
        },
        {
            "id": "ABL-02",
            "full_dialogue_context": "Our team spent 6 months optimizing the neural network latency down to 2ms. [SEP] The production deployment passed all stress tests with zero downtime.",
            "parent_context": "The production deployment passed all stress tests with zero downtime.",
            "response": "Oh brilliant job, that was truly a masterpiece of engineering!",
            "note": "Identical response words, but sincere situation; praise is genuine."
        },
        {
            "id": "ABL-03",
            "full_dialogue_context": "Is the weather nice for the beach today? [SEP] It is pouring rain with 60 mph gale force winds outside.",
            "parent_context": "It is pouring rain with 60 mph gale force winds outside.",
            "response": "Yeah, absolutely wonderful day for a swim.",
            "note": "Severe contrast between storm weather and outdoor activity response."
        }
    ]

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if os.path.exists(config.BEST_MODEL_DIR):
        tokenizer = AutoTokenizer.from_pretrained(config.BEST_MODEL_DIR)
        model = ContextAwareClassifier.from_pretrained(config.BEST_MODEL_DIR)
        model.to(device)
        print("Testing ablation on curated sample...")
        sample = get_curated_ablation_examples()[0]
        res = run_single_ablation(sample, model, tokenizer, device)
        for k, v in res.items():
            print(f"{k}: Pred={v['prediction']}, Prob={v['sarcasm_probability']}, Conf={v['confidence_percent']}")
    else:
        print("Model checkpoint not yet available. Run train.py first.")
