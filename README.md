# ◈ SUBTEXT

### **Context-Aware Sarcasm & Irony Detection**

<p align="center">

**Understand the words. Detect the contradiction. Reveal the SUBTEXT.**

</p>

<p align="center">

`AI / NLP` · `Transformers` · `Context Intelligence` · `Hackathon 03`

</p>

---

<div align="center">

### ⚡ CONTEXT CHANGES EVERYTHING

**“Oh brilliant job, that was truly a masterpiece of engineering!”**

**Without context:** Positive
**With context:** Sarcastic

</div>

---

## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## ◇ 01 · PROJECT

> **SUBTEXT** is a context-aware sarcasm and irony detection system designed to understand the difference between **what a person says** and **what they actually mean**.

Traditional sentiment models analyze the surface.

SUBTEXT analyzes the **conversation behind the surface**.

```text
                    ┌──────────────────┐
                    │    CONVERSATION  │
                    └────────┬─────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │  CONTEXT EXTRACTION │
                  └──────────┬──────────┘
                             │
                             ▼
             ┌──────────────────────────────┐
             │ CONTEXT + TARGET RESPONSE    │
             └──────────────┬───────────────┘
                            │
                            ▼
                 ╔═══════════════════╗
                 ║     DISTILBERT    ║
                 ║  CROSS-ENCODER    ║
                 ╚════════╤══════════╝
                          │
                          ▼
                 ┌─────────────────┐
                 │ CONTEXTUAL [CLS]│
                 └────────┬────────┘
                          │
                          ▼
                   CLASSIFICATION
                          │
                 ┌────────┴────────┐
                 ▼                 ▼
            SARCASM          NOT SARCASM
```

---

## ◈ THE CORE IDEA

Sarcasm is a **pragmatic phenomenon**.

The literal meaning of a sentence can be completely different from its intended meaning.

```text
┌─────────────────────────────────────────────┐
│ CONTEXT                                     │
│                                             │
│ “The payment system crashed for 4 hours.”  │
└──────────────────────┬──────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────┐
│ RESPONSE                                    │
│                                             │
│ “Brilliant job. Truly a masterpiece.”      │
└──────────────────────┬──────────────────────┘
                       │
                       ▼
                 ╔══════════════╗
                 ║ INCONGRUITY  ║
                 ╚══════╤═══════╝
                        │
                        ▼
                 ◆ SARCASM ◆
```

SUBTEXT learns this relationship rather than relying only on individual keywords.

---

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ◇ 02 · BENCHMARK

<div align="center">

| ◉ Metric       |     Result |
| -------------- | ---------: |
| **Macro F1**   | **0.7240** |
| **Sarcasm F1** | **0.7310** |
| **Accuracy**   | **0.7250** |

</div>

### Dataset

```text
╭──────────────────────────────────────────────╮
│                 DATASET                      │
├──────────────────────────────────────────────┤
│                                              │
│   TRAINING                  TEST             │
│                                              │
│   4,200                     2,000            │
│   examples                  examples         │
│                                              │
│   50% SARCASM               Unlabeled        │
│   50% NOT_SARCASM           IDs preserved    │
│                                              │
╰──────────────────────────────────────────────╯
```

### Class Distribution

```text
SARCASM

████████████████████████████████████  2,100
50.0%

NOT_SARCASM

████████████████████████████████████  2,100
50.0%
```

---

# ◇ 03 · WHY SUBTEXT?

### Traditional NLP

```text
WORDS
  ↓
FREQUENCY
  ↓
CLASSIFICATION
```

### SUBTEXT

```text
WORDS
  ↓
CONTEXT
  ↓
RELATIONSHIPS
  ↓
DISCOURSE
  ↓
INCONGRUITY
  ↓
INTENT
  ↓
CLASSIFICATION
```

The distinction is simple:

> **Sentiment asks: “Is this positive or negative?”**

> **SUBTEXT asks: “Does this statement make sense literally in this conversation?”**

---

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ◇ 04 · CONTEXT ENGINE

SUBTEXT evaluates three context strategies.

### `01` · RESPONSE_ONLY

```text
[CLS] response [SEP]
```

A context-free baseline.

---

### `02` · PARENT_RESPONSE

```text
[CLS] parent_context [SEP] response [SEP]
```

Adds the immediate conversational parent.

---

### `03` · FULL_CONTEXT_RESPONSE

```text
[CLS] full_dialogue_context [SEP] response [SEP]
```

Uses the complete available conversational history.

**Selected as the final strategy.**

---

## ◇ Intelligent Truncation

Long conversations cannot always fit into the Transformer context window.

SUBTEXT therefore uses a **left-truncation policy**.

```text
OLDER ─────────────────────────────── NEWER

[TURN 01]  ← REMOVE FIRST
[TURN 02]  ← REMOVE FIRST
[TURN 03]
[TURN 04]
[PARENT]   ← PRESERVE
[RESPONSE] ← PRESERVE
```

Maximum sequence length:

```text
128 SUBWORD TOKENS
```

The target response is prioritized and the most recent context is retained.

---

# ◇ 05 · BASELINE

Before introducing Transformers, SUBTEXT establishes a classical baseline.

### TF-IDF + Logistic Regression

| Context Strategy        | Macro F1 | Sarcasm F1 | Accuracy |
| ----------------------- | -------: | ---------: | -------: |
| Response Only           |   0.6594 |     0.6521 |   0.6595 |
| Parent + Response       |   0.6655 |     0.6675 |   0.6655 |
| Full Context + Response |   0.6351 |     0.6499 |   0.6357 |

### The Important Observation

Full context **hurts** the TF-IDF model.

Why?

Because bag-of-words representations primarily capture:

```text
WORD FREQUENCY
```

while sarcasm requires:

```text
WORD RELATIONSHIP
        +
CONTEXT
        +
DISCOURSE
        +
CONTRADICTION
```

This motivates the Transformer architecture.

---

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ◇ 06 · TRANSFORMER ENGINE

```text
                 INPUT
                   │
                   ▼
       ┌───────────────────────┐
       │ Context + Response    │
       └───────────┬───────────┘
                   │
                   ▼
       ┌───────────────────────┐
       │ Hugging Face Tokenizer│
       └───────────┬───────────┘
                   │
                   ▼
    [CLS] CONTEXT [SEP] RESPONSE [SEP]
                   │
                   ▼
       ╔═══════════════════════╗
       ║      DISTILBERT       ║
       ║                       ║
       ║  6 Transformer Layers ║
       ║  Bidirectional        ║
       ║  Self-Attention       ║
       ╚═══════════╤═══════════╝
                   │
                   ▼
            [CLS] REPRESENTATION
                   │
                   ▼
          ┌─────────────────┐
          │ Linear Head     │
          └────────┬────────┘
                   │
                   ▼
              SOFTMAX
                   │
          ┌────────┴────────┐
          ▼                 ▼
       SARCASM        NOT_SARCASM
```

### Why Cross-Encoder?

The context and response are processed **inside the same Transformer sequence**.

Therefore:

```text
Context Token
      ↕
Response Token
      ↕
Context Token
      ↕
Response Token
```

can directly interact through self-attention.

This enables the model to learn contextual contradictions.

---

# ◇ 07 · TRAINING CONFIGURATION

```text
╭─────────────────────────────────────────╮
│            MODEL CONFIGURATION           │
├─────────────────────────────────────────┤
│                                          │
│ Backbone        DistilBERT               │
│ Optimizer       AdamW                    │
│ Learning Rate   2 × 10⁻⁵                │
│ Weight Decay    0.01                     │
│ Batch Size      32                       │
│ Sequence       128 tokens                │
│ Split           80 / 20                  │
│ Warmup          10% steps                │
│ Selection       Validation Macro F1      │
│                                          │
╰─────────────────────────────────────────╯
```

The best checkpoint is selected according to **validation Macro F1**, matching the official benchmark objective.

---

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ◇ 08 · THRESHOLD INTELLIGENCE

The standard classification threshold is:

```text
P(SARCASM) ≥ 0.50
```

SUBTEXT instead searches for the threshold that maximizes **Macro F1**.

```text
0.20 ── 0.30 ── 0.40 ── 0.50 ── 0.60 ── 0.70 ── 0.85
  │       │       │       │       │       │       │
  └────────────────────────────────────────────────┘
                         │
                         ▼
                 BEST VALIDATION F1
                         │
                         ▼
              SAVED MODEL THRESHOLD
```

The selected threshold is stored in:

```text
models/best_model/model_metadata.json
```

and reused during test inference.

---

# ◇ 09 · RESULTS

### Model Comparison

```text
                         MACRO F1

TF-IDF / Response        ██████████████████████████░░░  0.6594

TF-IDF / Parent          ███████████████████████████░░  0.6655

TF-IDF / Full Context    █████████████████████████░░░░  0.6351

SUBTEXT Transformer      █████████████████████████████  0.7240
```

### Final SUBTEXT

| Metric         |      Score |
| -------------- | ---------: |
| **Macro F1**   | **0.7240** |
| **Sarcasm F1** | **0.7310** |
| **Accuracy**   | **0.7250** |

---

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ◇ 10 · CONTEXT ABLATION

One response.

Three levels of context.

One changing interpretation.

### Target

> **“Oh brilliant job, that was truly a masterpiece of engineering!”**

---

### `01` · Response Only

```text
Prediction → NOT_SARCASM

Confidence → 68%
```

The sentence contains strongly positive vocabulary.

---

### `02` · Parent Context

```text
“The entire payment system crashed for 4 hours.”
```

```text
Prediction → SARCASM

Confidence → 86%
```

The positive response contradicts the negative event.

---

### `03` · Full Conversation

```text
“Developer pushed code on Friday 5 PM...”
              ↓
“System crashed...”
              ↓
“Oh brilliant job...”
```

```text
Prediction → SARCASM

Confidence → 92%
```

Additional discourse strengthens the contextual contradiction.

---

# ◇ 11 · ERROR ANALYSIS

SUBTEXT also analyzes difficult examples instead of treating all errors equally.

### `01` · Keyword Traps

Words such as:

```text
“great”
“brilliant”
“obviously”
```

can appear in genuinely sincere statements.

---

### `02` · Sparse Context

Very short parent messages provide little information for establishing conversational contrast.

---

### `03` · Subtle Irony

Examples such as:

> **“What a surprise.”**

may contain almost no explicit sarcasm markers.

These cases require deeper semantic and pragmatic understanding.

---

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ◇ 12 · SUBMISSION PIPELINE

```text
TRAIN
  │
  ▼
PREPROCESS
  │
  ▼
BASELINE
  │
  ▼
TRAIN TRANSFORMER
  │
  ▼
VALIDATE
  │
  ▼
THRESHOLD TUNING
  │
  ▼
TEST INFERENCE
  │
  ▼
VALIDATE SUBMISSION
  │
  ▼
submission.csv
```

### Submission Contract

```text
outputs/submission.csv
```

```text
id,prediction
1,SARCASM
2,NOT_SARCASM
3,SARCASM
...
```

Requirements:

* Exactly **2 columns**
* Exactly **2,000 rows**
* IDs preserved
* Labels restricted to:

  * `SARCASM`
  * `NOT_SARCASM`

Validation:

```bash
python src/validator.py outputs/submission.csv
```

```text
[VALIDATION PASSED]
```

---

# ◇ 13 · ERROR → EVIDENCE → IMPROVEMENT

SUBTEXT is designed as an experimental system rather than a single model script.

```text
┌──────────────┐
│   DATASET    │
└──────┬───────┘
       ▼
┌──────────────┐
│      EDA     │
└──────┬───────┘
       ▼
┌──────────────┐
│   BASELINE   │
└──────┬───────┘
       ▼
┌──────────────┐
│  TRANSFORMER │
└──────┬───────┘
       ▼
┌──────────────┐
│  ABLATION    │
└──────┬───────┘
       ▼
┌──────────────┐
│ ERROR STUDY  │
└──────┬───────┘
       ▼
┌──────────────┐
│ CALIBRATION  │
└──────┬───────┘
       ▼
┌──────────────┐
│   INFERENCE  │
└──────────────┘
```

Every stage produces evidence that informs the next.

---

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ◇ 14 · PROJECT ARCHITECTURE

```text
SUBTEXT/
│
├── ◇ data/
│   ├── train.csv
│   └── test.csv
│
├── ◇ notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baseline.ipynb
│   ├── 03_training.ipynb
│   └── 04_evaluation.ipynb
│
├── ◇ src/
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── inference.py
│   ├── ablation.py
│   ├── evidence.py
│   ├── validator.py
│   └── api.py
│
├── ◇ models/
│   └── best_model/
│
├── ◇ outputs/
│   ├── metrics.json
│   ├── validation_predictions.csv
│   ├── test_predictions.csv
│   └── submission.csv
│
├── ◇ reports/
│   ├── eda_report.md
│   ├── error_analysis.md
│   ├── eda_distributions.png
│   └── experiment_results.csv
│
├── requirements.txt
└── README.md
```

---

# ◇ 15 · QUICK START

### Install

```bash
pip install -r requirements.txt
```

### Run EDA

```bash
python src/eda.py
```

### Run Baselines

```bash
python src/baseline.py
```

### Train SUBTEXT

```bash
python src/train.py
```

### Generate Predictions

```bash
python src/inference.py
```

### Validate Submission

```bash
python src/validator.py outputs/submission.csv
```

---

# ◇ 16 · INTERACTIVE DEMO

SUBTEXT can also be exposed through the prediction API.

### Start API

```bash
python src/api.py
```

### Start Frontend

```bash
cd ../subtext-app
npm run dev
```

Then open:

```text
http://localhost:5173
```

---

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ◇ 17 · TECHNOLOGY STACK

<div align="center">

`Python` · `PyTorch` · `Hugging Face` · `DistilBERT`
`Scikit-learn` · `FastAPI` · `React` · `Vite`

</div>

---

# ◇ 18 · THE TAKEAWAY

> ### **Sarcasm isn't always in the sentence.**
>
> ### **Sometimes, it's between the sentences.**

SUBTEXT models that space.

```text
LITERAL WORDS
      +
CONVERSATIONAL CONTEXT
      +
TRANSFORMER ATTENTION
      +
PRAGMATIC INCONGRUITY
      ↓
     SUBTEXT
```

---

<div align="center">

## ◈ SUBTEXT

**Context-Aware Sarcasm & Irony Detection**

### Built by Team Cyber Spartans

**P. Dinesh Kumar · Ch. Karthik · M. Sai Koushik Kumar · D. Harshith**

---

`Hackathon Challenge 03`
`Department of AI & Data Science`

**Understand the words. Detect the meaning behind them.**

</div>
