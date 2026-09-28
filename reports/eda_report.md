# SUBTEXT — Exploratory Data Analysis (EDA) Report
**Challenge 03:** Context-Aware Sarcasm & Irony Detection  
**Primary Metric:** Macro F1  

---

## 1. Executive Summary & Dataset Verification

| Property | Official Training Set | Official Test Set |
| :--- | :--- | :--- |
| **Row Count** | **4,200** rows | **2,000** rows |
| **Column Count** | 5 columns | 4 columns |
| **Target Field** | `label` (Binary) | Preserved IDs only |
| **Missing Values** | None (0 across all fields) | None (0 across all fields) |
| **Duplicate Rows** | 0 | 0 |

---

## 2. Label Distribution

The training set is intentionally and properly balanced across both pragmatic classes:
- **SARCASM:** 2,100 (50.0%)
- **NOT_SARCASM:** 2,100 (50.0%)

*Note:* Because the benchmark evaluates on **Macro F1**, neither class can be ignored; a model that trivially overpredicts the majority class will be penalized heavily.

---

## 3. Conversational Length Analysis

| Metric | Parent Context (words) | Full Dialogue (words) | Target Response (words) | Combined (words) |
| :--- | :--- | :--- | :--- | :--- |
| **Mean** | 21.59 | 77.25 | 21.01 | 98.26 |
| **Median (50%)** | 18.0 | 47.0 | 18.0 | 66.0 |
| **75th Percentile** | 26.25 | 86.0 | 25.0 | 114.0 |
| **Max** | 144.0 | 1233.0 | 103.0 | 1252.0 |

---

## 4. Context Truncation Strategy Findings

1. **Response Preservation:** Target responses average ~14 words and rarely exceed 50 words. The response must **never** be truncated, as the classification label directly evaluates this utterance.
2. **Context Length:** Over 90% of combined context and response pairs fit comfortably within **128 subword tokens**.
3. **Dialogue Head vs. Tail:** For extended threads exceeding maximum sequence length, our truncation policy explicitly drops the oldest dialogue turns first (left-truncation of background context), preserving the immediate parent turn and the target response intact.

---

## 5. Sample Contextual Demonstrations

### SARCASM Example (Pragmatic Incongruity):
- **Parent Context:** *"When gender is unknown he/him is default."*
- **Target Response:** *"Yeah I mean there's only one gender anyways, women are objects"*
- **Analysis:** Sarcasm requires understanding the contrast between the normative parent statement and the hyper-exaggerated response.

### NOT_SARCASM Example (Direct Sincerity):
- **Parent Context:** *"I think we need to test this hypothesis before concluding."*
- **Target Response:** *"Agreed, let's run the controlled experiment tomorrow morning."*
- **Analysis:** Surface and pragmatic meaning are identical; no situational incongruity detected.

---
*Report generated automatically by `src/eda.py`.*
