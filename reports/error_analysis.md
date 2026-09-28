# SUBTEXT — Model Error Analysis & Technical Defense Report
**Challenge 03:** Context-Aware Sarcasm & Irony Detection
**Threshold Used:** 0.50

---
## 1. Confusion Matrix Breakdown
- **True Positives (TP - Correct Sarcasm):** 335
- **True Negatives (TN - Correct Sincere):** 273
- **False Positives (FP - Sincere misclassified as Sarcasm):** 155
- **False Negatives (FN - Sarcasm misclassified as Sincere):** 77

---
## 2. Hard-Case Analysis (Lowest Confidence / Pragmatic Ambiguity)
The examples below exhibit high pragmatic ambiguity, subtle ironic incongruity, or deceptive sentiment lexical cues:

### Case #1 (ID: SARC_TR_01920)
- **Parent Context:** *"PP literally kills little babies that haven't even been born yet :/"*
- **Target Response:** *"Prove that it is wrong thing to do."*
- **Ground Truth:** `NOT_SARCASM` | **Model Prediction:** `NOT_SARCASM`
- **Prediction Confidence:** `50.1%` (Sarcasm Prob: `0.499`)
- **Context Evidence / Diagnosis:** Boundary case near the 0.50 decision threshold

### Case #2 (ID: SARC_TR_04129)
- **Parent Context:** *"it's not so quiet if it's immediately known"*
- **Target Response:** *"At least nobody post it on Reddit, it's just a bunch of blabbermouths over there"*
- **Ground Truth:** `SARCASM` | **Model Prediction:** `SARCASM`
- **Prediction Confidence:** `50.1%` (Sarcasm Prob: `0.501`)
- **Context Evidence / Diagnosis:** Boundary case near the 0.50 decision threshold

### Case #3 (ID: SARC_TR_02878)
- **Parent Context:** *""Waaahhh I only pip 95% of my gamesss, killer is so hard :(""*
- **Target Response:** *"You don't even get the point of this post, please go be a shitty troll somewhere else."*
- **Ground Truth:** `NOT_SARCASM` | **Model Prediction:** `NOT_SARCASM`
- **Prediction Confidence:** `50.2%` (Sarcasm Prob: `0.498`)
- **Context Evidence / Diagnosis:** Boundary case near the 0.50 decision threshold

### Case #4 (ID: SARC_TR_02990)
- **Parent Context:** *"To be fair quite a few on those list are blatantly anti-Semitic (or trolls)."*
- **Target Response:** *"I don't know, I bet hitlerwasright is a charming person."*
- **Ground Truth:** `NOT_SARCASM` | **Model Prediction:** `NOT_SARCASM`
- **Prediction Confidence:** `50.3%` (Sarcasm Prob: `0.497`)
- **Context Evidence / Diagnosis:** Boundary case near the 0.50 decision threshold

### Case #5 (ID: SARC_TR_02019)
- **Parent Context:** *"To all the commenters shitting on these people for donating money, what are you doing to help?"*
- **Target Response:** *"Avoiding organized religion and the inevitable minority of zealots that will twist any religious worldview into an excuse for violence?"*
- **Ground Truth:** `NOT_SARCASM` | **Model Prediction:** `NOT_SARCASM`
- **Prediction Confidence:** `50.3%` (Sarcasm Prob: `0.497`)
- **Context Evidence / Diagnosis:** Punctuation-driven ambiguity; surface sarcasm vs genuine enthusiasm

### Case #6 (ID: SARC_TR_01577)
- **Parent Context:** *"One of these things is not like the others."*
- **Target Response:** *"Is it reynad, due to his prowess at magic and hearthstone?"*
- **Ground Truth:** `NOT_SARCASM` | **Model Prediction:** `NOT_SARCASM`
- **Prediction Confidence:** `50.4%` (Sarcasm Prob: `0.496`)
- **Context Evidence / Diagnosis:** Punctuation-driven ambiguity; surface sarcasm vs genuine enthusiasm

### Case #7 (ID: SARC_TR_03989)
- **Parent Context:** *"How much did she make on aquaman?"*
- **Target Response:** *"They should use a neural network to put his face over hers throughout the movie."*
- **Ground Truth:** `NOT_SARCASM` | **Model Prediction:** `SARCASM`
- **Prediction Confidence:** `50.4%` (Sarcasm Prob: `0.504`)
- **Context Evidence / Diagnosis:** Boundary case near the 0.50 decision threshold

### Case #8 (ID: SARC_TR_04026)
- **Parent Context:** *"Don't forget to thank Obama (for this mess caused by Obama)! "*
- **Target Response:** *"I heard Obama was golfing during hurricanes Irma and Harvey (as always) "*
- **Ground Truth:** `SARCASM` | **Model Prediction:** `NOT_SARCASM`
- **Prediction Confidence:** `50.6%` (Sarcasm Prob: `0.494`)
- **Context Evidence / Diagnosis:** Boundary case near the 0.50 decision threshold

### Case #9 (ID: SARC_TR_01329)
- **Parent Context:** *"do you know what sarcasm is."*
- **Target Response:** *"Do you know what a question mark is?"*
- **Ground Truth:** `SARCASM` | **Model Prediction:** `NOT_SARCASM`
- **Prediction Confidence:** `50.9%` (Sarcasm Prob: `0.491`)
- **Context Evidence / Diagnosis:** Punctuation-driven ambiguity; surface sarcasm vs genuine enthusiasm

### Case #10 (ID: SARC_TR_01932)
- **Parent Context:** *"Correction: @shidavidi reporting it's single A Dunedin not AAA Buffalo"*
- **Target Response:** *"Does that mean he is going by Aron now?"*
- **Ground Truth:** `NOT_SARCASM` | **Model Prediction:** `NOT_SARCASM`
- **Prediction Confidence:** `51.0%` (Sarcasm Prob: `0.490`)
- **Context Evidence / Diagnosis:** Punctuation-driven ambiguity; surface sarcasm vs genuine enthusiasm

---
## 3. Key Error Patterns & Mitigation for 2-Minute Defense
1. **Shallow Lexical Cues vs. Pragmatic Incongruity:** Bag-of-words baselines trigger false positives on words like *'obviously'*, *'definitely'*, or *'great'*. The transformer encoder cross-attends the parent context to distinguish literal affirmations from sarcastic incongruity.
2. **Sparse Context Vulnerability:** When parent context is one or two words, contextual incongruity signals are attenuated, leading to lower model confidence.
3. **Tone Polarization:** Invertive sarcasm where positive words express negative evaluation remains harder than hyperbolic exaggeration.
