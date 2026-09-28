import os
import sys
import json
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from transformers import AutoTokenizer

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from model import ContextAwareClassifier
from preprocessing import build_context_pair
from ablation import run_single_ablation, get_curated_ablation_examples
from evidence import get_hard_cases, extract_context_evidence

app = FastAPI(
    title="SUBTEXT — Context-Aware Sarcasm Detection API",
    description="Official Hackathon ML Inference & Evidence Service",
    version="1.0.0"
)

# Enable CORS for React/Vite frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model state
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
tokenizer = None
threshold = 0.50
metadata = {}

def get_model():
    global model, tokenizer, threshold, metadata
    if model is None:
        model_dir = config.BEST_MODEL_DIR
        if not os.path.exists(os.path.join(model_dir, "model.safetensors")) and not os.path.exists(os.path.join(model_dir, "pytorch_model.bin")):
            # Fall back to base pretrained model if best checkpoint not yet saved
            model_dir = config.MODEL_NAME

        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = ContextAwareClassifier.from_pretrained(model_dir, num_labels=2)
        model.to(device)
        model.eval()

        meta_path = os.path.join(config.BEST_MODEL_DIR, "model_metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            threshold = metadata.get("selected_threshold", 0.50)
    return model, tokenizer, threshold

class PredictRequest(BaseModel):
    parent_context: Optional[str] = ""
    full_dialogue_context: Optional[str] = ""
    response: str

class PredictResponse(BaseModel):
    prediction: str
    confidence: float
    confidence_percent: str
    sarcasm_probability: float
    model: str = "DeBERTa-v3-base"
    context_used: Optional[str] = ""
    context_evidence: Dict[str, Any]

@app.on_event("startup")
def startup_event():
    try:
        get_model()
    except Exception as e:
        print(f"Model startup warning (will lazy load): {e}")

@app.get("/")
def health_check():
    return {
        "status": "online",
        "service": "SUBTEXT Sarcasm Detection Engine",
        "primary_metric": "Macro F1",
        "device": str(device),
        "threshold": threshold
    }

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if not req.response.strip():
        raise HTTPException(status_code=400, detail="Response text cannot be empty.")
    
    m, tok, th = get_model()
    row = {
        "parent_context": req.parent_context or "",
        "full_dialogue_context": req.full_dialogue_context or "",
        "response": req.response
    }

    context, response = build_context_pair(row, strategy=config.DEFAULT_CONTEXT_STRATEGY)
    
    if not context:
        enc = tok(text=response, max_length=config.MAX_SEQ_LENGTH, padding="max_length", truncation=True, return_tensors="pt")
    else:
        enc = tok(text=context, text_pair=response, max_length=config.MAX_SEQ_LENGTH, padding="max_length", truncation=True, return_tensors="pt")

    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    with torch.no_grad():
        out = m(input_ids=input_ids, attention_mask=attention_mask)
        sarcasm_prob = float(out["probs"][0, 1].item())

    pred_id = int(sarcasm_prob >= th)
    pred_label = config.ID2LABEL[pred_id]
    conf = sarcasm_prob if pred_id == 1 else (1.0 - sarcasm_prob)

    evidence = extract_context_evidence(
        parent_context=req.parent_context or "",
        full_dialogue_context=req.full_dialogue_context or "",
        response=req.response
    )

    return {
        "prediction": pred_label,
        "confidence": round(conf, 4),
        "confidence_percent": f"{conf * 100:.1f}%",
        "sarcasm_probability": round(sarcasm_prob, 4),
        "model": "DeBERTa-v3-base",
        "context_used": context if context else "(None - Response Isolated)",
        "context_evidence": evidence
    }

@app.post("/compare-context")
def compare_context(req: PredictRequest):
    if not req.response.strip():
        raise HTTPException(status_code=400, detail="Response text cannot be empty.")
    
    m, tok, th = get_model()
    row = {
        "parent_context": req.parent_context or "",
        "full_dialogue_context": req.full_dialogue_context or "",
        "response": req.response
    }
    ablation_results = run_single_ablation(row, m, tok, device, threshold=th)
    return {
        "target_response": req.response,
        "strategies": ablation_results
    }

@app.get("/hard-cases")
def hard_cases():
    return {
        "cases": get_hard_cases(top_n=8),
        "curated_ablation_samples": get_curated_ablation_examples()
    }

@app.get("/metrics")
def get_metrics():
    meta_path = os.path.join(config.BEST_MODEL_DIR, "model_metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    elif os.path.exists(config.EXPERIMENT_RESULTS_FILE):
        df = pd.read_csv(config.EXPERIMENT_RESULTS_FILE)
        return {"experiments": df.to_dict(orient="records")}
    return {"status": "Training in progress"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
