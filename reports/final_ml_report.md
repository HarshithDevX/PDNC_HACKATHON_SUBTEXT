# SUBTEXT — Final Machine Learning & NLP Evaluation Report

**Hackathon Challenge 03:** Context-Aware Sarcasm & Irony Detection  
**Primary Metric:** Macro F1  
**Secondary Metrics:** Sarcasm F1, Classification Accuracy  
**Model Architecture:** `microsoft/deberta-v3-base` (Strictly Single Backbone)  
**Execution Timestamp:** 2026-09-28 21:19:25 IST  

---

## 1. Problem Statement
Sarcasm detection requires resolving the pragmatic incongruity between an utterance's literal surface semantics and its intended communicative meaning. In multi-turn dialogue, understanding whether an utterance is sarcastic often depends on preceding discourse context. This project experimentally evaluates three distinct context depth configurations using the `microsoft/deberta-v3-base` transformer cross-encoder on the official hackathon benchmark.

---

## 2. Official Dataset Specification
The official dataset was strictly used for all modeling, validation, and testing:
- **Training Set (`data/official/train.csv`):** **4,200** rows
  - Columns: `id`, `parent_context`, `full_dialogue_context`, `response`, `label`, `label_id`
  - Split: Stratified 80/20 train/validation split (3,360 train / 840 validation, random seed 42)
  - Label balance in validation: SARCASM = 412, NOT_SARCASM = 428
- **Test Set (`data/official/test.csv`):** **2,000** rows
  - Columns: `id`, `parent_context`, `full_dialogue_context`, `response`
  - Preservation: 100% 1-to-1 match of test IDs in original sequence

---

## 3. Model Architecture & Training Design
- **Backbone Architecture:** `microsoft/deberta-v3-base` (DeBERTaV2ForSequenceClassification)
- **Precision:** Full float32 on CUDA (ensuring numerical stability across attention layers)
- **Optimizer:** AdamW (`lr=2e-5`, `eps=1e-7`, `weight_decay=0.01`)
- **Learning Rate Schedule:** Linear warmup over first 10% of training steps with linear decay
- **Sequence Length:** 128 subword tokens
- **Batch Size:** 16 (per device)
- **Hardware Acceleration:** NVIDIA GeForce RTX 4060 Laptop GPU (8,188 MiB VRAM) via PyTorch 2.6.0+cu124

---

## 4. Context Configurations Evaluated
Three context representations were evaluated across identical training configurations:
- **Experiment A (Response Only):** `[CLS] response [SEP]`  
  *Context-free baseline quantifying intrinsic lexical cues.*
- **Experiment B (Parent Context + Response):** `[CLS] parent_context [SEP] response [SEP]`  
  *Conditioning on the immediate preceding conversational turn.*
- **Experiment C (Full Dialogue Context + Response):** `[CLS] full_dialogue_context [SEP] response [SEP]`  
  *Conditioning on the full multi-turn conversational history with left-truncation preserving immediate turns.*

---

## 5. Measured Validation Results (A / B / C Comparison)

All metrics recorded on the identical 840-sample stratified validation split:

| Experiment | Context Configuration | Best Epoch | Macro F1 (Primary) | Sarcasm F1 | Accuracy | Checkpoint Directory |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **A** | **Response Only** | **2** | **0.7223** | **0.7428** | **0.7238** | `models/checkpoint_exp_a` |
| **B** | Parent + Response | 3 | 0.7094 | 0.7039 | 0.7095 | `models/checkpoint_exp_b` |
| **C** | Full Dialogue + Response | 2 | 0.7115 | 0.7225 | 0.7119 | `models/checkpoint_exp_c` |

*Factual Observation:* Experiment A achieved the highest validation Macro F1 (0.7223), followed by Experiment C (0.7115) and Experiment B (0.7094).

---

## 6. Final Model Selection
- **Selected Configuration:** **Experiment A (`Response Only`)**
- **Selection Criterion:** Strict highest validation Macro F1 (0.7223)
- **Final Checkpoint Path:** `models/best_model` (promoted directly from `models/checkpoint_exp_a` without retraining)
- **Metadata File:** `models/best_model/model_metadata.json`

---

## 7. Official Test Submission & Validator Verification
- **Submission File Path:** `outputs/submission.csv`
- **Detailed Predictions Path:** `outputs/test_predictions.csv`
- **Row Count:** Exactly **2,000** test rows
- **Columns:** Strictly `id,prediction`
- **Prediction Label Mapping:** `1` -> `SARCASM` (1,180 rows), `0` -> `NOT_SARCASM` (820 rows)
- **ID Alignment:** 100% sequential alignment with `data/official/test.csv` (IDs preserved, no missing/extra rows)
- **Official Validator Output:** **PASS**
  - Project Validator (`src/validator.py`): `[VALIDATION PASSED] All checks succeeded successfully!`
  - Organizer Validator (`data/official/validator.py`): `[STATUS] Passed validation! Ready for evaluation (2000 rows).`

---

## 8. Error Analysis (Validation Hard Cases & Failure Modes)

Based on actual validation predictions from the winning model:
- **Confusion Matrix:**
  - True Positives (TP): 335
  - True Negatives (TN): 273
  - False Positives (FP): 155
  - False Negatives (FN): 77

### Key Diagnostic Failure Modes:
1. **Decision Boundary Sensitivity (49.5% - 50.5% Confidence):**  
   Near-threshold utterances exhibiting ambiguous tone (e.g. sarcastic hyperbole phrased as dry factual questions) represent the primary source of classification errors.
2. **Lexical Irony without Surface Markers:**  
   Utterances that contain no overt affective adjectives (e.g., *"Prove that it is wrong thing to do."*) are difficult to distinguish from sincere discourse without speaker tone cues.
3. **Punctuation Nuance:**  
   Statements ending with question marks or ellipses often express rhetorical skepticism rather than sincere inquiry, causing misclassification if the model treats questions as sincere.
