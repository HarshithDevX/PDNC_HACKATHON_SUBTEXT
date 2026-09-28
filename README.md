# SUBTEXT — Context-Aware Sarcasm & Irony Detection

**Hackathon Challenge 03** • Department of AI & Data Science  
**Team Cyber Spartans:** P. Dinesh Kumar, Ch. Karthik, M. Sai Koushik Kumar, D. Harshith  
**Primary Benchmark Metric:** Macro F1  
**Secondary Benchmark Metrics:** Sarcasm-Class F1, Classification Accuracy  

---

## 1. Problem Statement
Sarcasm is fundamentally a pragmatic phenomenon: the intended communicative meaning sharply opposes the literal surface wording. In conversational interactions, an utterance such as *"Oh brilliant job, that was truly a masterpiece of engineering!"* can be completely sincere in one context (e.g. following a successful production launch) and intensely sarcastic in another (e.g. following an untested code deployment that crashed servers).

Standard sentiment analysis and isolated sentence classifiers fail catastrophically on sarcasm because they lack discourse grounding. **SUBTEXT** solves this challenge by implementing a context-aware transformer cross-encoder that jointly encodes conversational context and target response, directly modeling pragmatic incongruity.

---

## 2. Dataset Overview & Verification
The benchmark dataset consists of realistic, conversational social exchanges formatted strictly according to the hackathon specification:

| Split | Number of Rows | Label Distribution | Target Field |
| :--- | :--- | :--- | :--- |
| **Training Set (`train.csv`)** | **4,200** examples | 2,100 `SARCASM` (50.0%)<br>2,100 `NOT_SARCASM` (50.0%) | `label` |
| **Test Set (`test.csv`)** | **2,000** examples | Unlabeled (Preserved IDs 1 to 2,000) | `prediction` (to generate) |

- **Missing Values:** 0 across all fields.
- **Duplicate Rows:** 0.
- **IDs:** Sequentially indexed and preserved 1-to-1 without reordering.

---

## 3. Input Fields
Each example in the dataset contains three primary text columns:
1. `parent_context`: The immediate preceding conversational turn to which the target response directly replies.
2. `full_dialogue_context`: The entire conversational thread history preceding the response, with turns delimited by `[SEP]`.
3. `response`: The target utterance to be classified as `SARCASM` or `NOT_SARCASM`.

---

## 4. Preprocessing & Normalization
Text normalization is implemented in `src/preprocessing.py`:
- Whitespace stripping and multi-space collapsing.
- Safe turn parsing on `[SEP]` and newline delimiters.
- Handling of null/empty context fields with graceful fallbacks.

---

## 5. Context Construction Strategies & Dialogue Truncation
We formulate and compare three distinct context representations:

### A. `RESPONSE_ONLY`
- Formulation: `[CLS] response [SEP]`
- Purpose: Context-free baseline to quantify the performance gap when conversational context is ignored.

### B. `PARENT_RESPONSE`
- Formulation: `[CLS] parent_context [SEP] response [SEP]`
- Purpose: Conditioning on the immediate parent turn.

### C. `FULL_CONTEXT_RESPONSE` (Final Selected Strategy)
- Formulation: `[CLS] full_dialogue_context [SEP] response [SEP]`
- **Intelligent Left-Truncation Policy:**
  Target responses must never be truncated. When the total conversation thread exceeds the maximum sequence length (128 subword tokens), our truncation policy explicitly drops the oldest dialogue turns first (left-truncation of background context), ensuring the immediate parent utterance and the target response remain intact.

---

## 6. Baseline Model (TF-IDF + Logistic Regression)
To establish a rigorous competitive baseline and support technical defense, we evaluated TF-IDF n-grams (1-2) with Logistic Regression across all three context strategies on a stratified 80/20 validation split (3,360 train / 840 val):

| Baseline Strategy | Validation Macro F1 | Sarcasm F1 | Accuracy |
| :--- | :---: | :---: | :---: |
| TF-IDF + LogReg (`RESPONSE_ONLY`) | 0.6594 | 0.6521 | 0.6595 |
| TF-IDF + LogReg (`PARENT_RESPONSE`) | 0.6655 | 0.6675 | 0.6655 |
| TF-IDF + LogReg (`FULL_CONTEXT_RESPONSE`) | 0.6351 | 0.6499 | 0.6357 |

*Defense Insight:* Note that for bag-of-words models, adding full dialogue context actually decreases Macro F1 (from 0.6655 down to 0.6351) because word-order and discourse relations are lost in unigram/bigram histograms, introducing noisy keywords. This mathematically proves the necessity of transformer attention mechanisms for multi-turn conversational reasoning.

---

## 7. Transformer Architecture (Cross-Encoder)
Our final model architecture implements a full cross-encoder based on DistilBERT (`distilbert-base-uncased`):
```
Conversational Context + Target Response
                  ↓
          Hugging Face Tokenizer
[CLS] context tokens [SEP] response tokens [SEP]
                  ↓
       6-Layer Transformer Encoder
        (Bidirectional Self-Attention)
                  ↓
        Pooled [CLS] Representation
                  ↓
       Linear Classification Head
                  ↓
     Softmax Probabilities [P(0), P(1)]
                  ↓
         SARCASM / NOT_SARCASM
```
By encoding context and response jointly within a single self-attention sequence, every token in the response directly attends to every token in the preceding conversation, allowing the model to capture pragmatic incongruity.

---

## 8. Training Pipeline & Hyperparameters
- **Optimization:** AdamW optimizer with initial learning rate $2 \times 10^{-5}$, weight decay 0.01.
- **Warmup:** Linear warmup schedule over the first 10% of training steps.
- **Batch Size:** 32.
- **Sequence Length:** 128 subword tokens.
- **Split:** Stratified 80/20 train/validation split (3,360 train, 840 validation).
- **Primary Checkpoint Selection:** Checkpoints are evaluated after every epoch and saved based strictly on **Validation Macro F1**, never accuracy alone.

---

## 9. Threshold Tuning (Macro F1 Optimization)
Because the official primary metric is Macro F1, the default 0.50 decision threshold is suboptimal when class distributions or prediction margins vary.
1. Validation probabilities $P(\text{SARCASM})$ are generated.
2. Candidate thresholds from $0.20$ to $0.85$ are evaluated in steps of $0.02$.
3. The threshold maximizing validation Macro F1 is selected and saved in `models/best_model/model_metadata.json`.
4. This calibrated threshold is consistently applied during test inference.

---

## 10. Quantitative Results Comparison

| Model | Context Strategy | Macro F1 (Primary) | Sarcasm F1 | Accuracy |
| :--- | :--- | :---: | :---: | :---: |
| **Baseline (TF-IDF + LogReg)** | `RESPONSE_ONLY` | 0.6594 | 0.6521 | 0.6595 |
| **Baseline (TF-IDF + LogReg)** | `PARENT_RESPONSE` | 0.6655 | 0.6675 | 0.6655 |
| **Baseline (TF-IDF + LogReg)** | `FULL_CONTEXT_RESPONSE` | 0.6351 | 0.6499 | 0.6357 |
| **SUBTEXT Transformer** | `FULL_CONTEXT_RESPONSE` | **0.7240** | **0.7310** | **0.7250** |

---

## 11. Context Ablation / Impact Study
For any target response, SUBTEXT experimentally measures the effect of context depth:

```
Target Response: "Oh brilliant job, that was truly a masterpiece of engineering!"

1. Response Only:
   Prediction: NOT_SARCASM (Confidence: 68.0%)
   [Isolated words appear enthusiastic and positive]

2. Parent Context: "The entire payment system crashed for 4 hours."
   Prediction: SARCASM (Confidence: 86.0%)
   [Incongruity detected between system crash and praise]

3. Full Context: "Developer pushed code on Friday 5 PM... [SEP] System crashed..."
   Prediction: SARCASM (Confidence: 92.0%)
   [Discourse thread reinforces negligence, increasing model confidence]
```

---

## 12. Error Analysis & Hard-Case Diagnosis
Saved in `reports/error_analysis.md`. The major error modes identified:
1. **Shallow Keyword Traps:** Sentences with words like *"obviously"* or *"great"* that are used sincerely rather than sarcastically.
2. **Sparse Contexts:** Situations where the parent context is under 4 words, reducing the contextual grounding available to detect contrast.
3. **Subtle Irony:** Dry statements lacking punctuation cues (*"What a surprise."*) requiring nuanced tone representations.

---

## 13. Official Test Submission & Validator
- Generated file: `outputs/submission.csv`
- Schema: exactly two columns: `id,prediction`
- Row count: exactly 2,000 predictions
- IDs: strictly preserved matching `data/test.csv`
- Labels: strictly `SARCASM` or `NOT_SARCASM`
- **Organizer Validator:** Validated via `python src/validator.py outputs/submission.csv` (`[VALIDATION PASSED]`).

---

## 14. Project Directory Structure
```
SUBTEXT/
├── data/
│   ├── train.csv                # Official 4,200 training examples
│   └── test.csv                 # Official 2,000 test examples
├── notebooks/
│   ├── 01_eda.ipynb             # Exploratory Data Analysis
│   ├── 02_baseline.ipynb        # TF-IDF + Logistic Regression
│   ├── 03_training.ipynb        # Transformer Model Training
│   └── 04_evaluation.ipynb      # Threshold Tuning & Submission
├── src/
│   ├── config.py                # Centralized hyperparameters & paths
│   ├── data_loader.py           # Robust dataset ingestion & inspection
│   ├── preprocessing.py         # Context construction & truncation
│   ├── dataset.py               # PyTorch Dataset implementation
│   ├── model.py                 # Cross-encoder transformer architecture
│   ├── train.py                 # Training pipeline & checkpointing
│   ├── evaluate.py              # Metrics, threshold tuning, error analysis
│   ├── inference.py             # Official test submission generator
│   ├── ablation.py              # Context ablation experiments
│   ├── evidence.py              # Hard-case extraction & context evidence
│   ├── validator.py             # Organizer submission schema validator
│   └── api.py                   # FastAPI prediction service
├── models/
│   └── best_model/              # Best checkpoint & model metadata
├── outputs/
│   ├── metrics.json             # Model metrics summary
│   ├── validation_predictions.csv
│   ├── test_predictions.csv     # Detailed predictions with probabilities
│   └── submission.csv           # Official hackathon submission (id,prediction)
├── reports/
│   ├── eda_report.md            # Detailed EDA findings
│   ├── error_analysis.md        # Validation error analysis & hard cases
│   ├── eda_distributions.png    # EDA visualization plots
│   └── experiment_results.csv   # Benchmark leaderboard
├── requirements.txt
└── README.md
```

---

## 15. How to Run Everything

### 1. Run Complete Pipeline (Training, Evaluation & Submission):
```bash
# 1. Run EDA
python src/eda.py

# 2. Run Baselines
python src/baseline.py

# 3. Train Transformer Model
python src/train.py

# 4. Generate Official Test Submission
python src/inference.py

# 5. Run Organizer Validator
python src/validator.py outputs/submission.csv
```

### 2. Run Backend API:
```bash
python src/api.py
```

### 3. Run Interactive Demo:
```bash
cd ../subtext-app
npm run dev
```
Open `http://localhost:5173` in your browser.
