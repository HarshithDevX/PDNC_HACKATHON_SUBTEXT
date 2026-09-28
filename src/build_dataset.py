import urllib.request
import json
import pandas as pd
import numpy as np
import os

data_dir = r"C:\Users\donda\.gemini\antigravity-ide\scratch\SUBTEXT\data"
os.makedirs(data_dir, exist_ok=True)

print("Fetching FigLang shared task dialogue dataset...")
urls = [
    "https://huggingface.co/datasets/tasksource/figlang2020-sarcasm/raw/main/sarcasm_detection_shared_task_reddit_training.jsonl",
    "https://huggingface.co/datasets/tasksource/figlang2020-sarcasm/raw/main/sarcasm_detection_shared_task_twitter_training.jsonl"
]

all_records = []
for url in urls:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        for line in resp:
            line_str = line.decode("utf-8").strip()
            if not line_str:
                continue
            item = json.loads(line_str)
            raw_context = item.get("context", [])
            if isinstance(raw_context, list):
                parent = raw_context[-1].strip() if len(raw_context) > 0 else ""
                full_ctx = " [SEP] ".join([c.strip() for c in raw_context if c.strip()])
            else:
                parent = str(raw_context).strip()
                full_ctx = parent
            
            resp_text = str(item.get("response", "")).strip()
            label = str(item.get("label", "")).strip().upper()
            if label not in ["SARCASM", "NOT_SARCASM"]:
                label = "SARCASM" if label in ["1", 1, "TRUE", True] else "NOT_SARCASM"
            
            if resp_text and (parent or full_ctx):
                all_records.append({
                    "parent_context": parent,
                    "full_dialogue_context": full_ctx,
                    "response": resp_text,
                    "label": label
                })

df_all = pd.DataFrame(all_records)
print(f"Total raw candidate pool: {len(df_all)}")
print("Class breakdown in pool:")
print(df_all["label"].value_counts())

# Stratified sampling to create exact 4,200 train and 2,000 test
np.random.seed(42)
sarcasm_pool = df_all[df_all["label"] == "SARCASM"].sample(frac=1.0, random_state=42).reset_index(drop=True)
not_sarcasm_pool = df_all[df_all["label"] == "NOT_SARCASM"].sample(frac=1.0, random_state=42).reset_index(drop=True)

# 4,200 train: 2,100 SARCASM, 2,100 NOT_SARCASM
train_sarc = sarcasm_pool.iloc[:2100]
train_not_sarc = not_sarcasm_pool.iloc[:2100]
df_train = pd.concat([train_sarc, train_not_sarc]).sample(frac=1.0, random_state=42).reset_index(drop=True)
df_train.insert(0, "id", range(1, len(df_train) + 1))

# 2,000 test: 1,000 SARCASM, 1,000 NOT_SARCASM
test_sarc = sarcasm_pool.iloc[2100:3100]
test_not_sarc = not_sarcasm_pool.iloc[2100:3100]
df_test_full = pd.concat([test_sarc, test_not_sarc]).sample(frac=1.0, random_state=42).reset_index(drop=True)
df_test_full.insert(0, "id", range(1, len(df_test_full) + 1))

# Test set for participants does NOT contain label
df_test = df_test_full[["id", "parent_context", "full_dialogue_context", "response"]].copy()

# Save official train and test CSVs
train_path = os.path.join(data_dir, "train.csv")
test_path = os.path.join(data_dir, "test.csv")
test_truth_path = os.path.join(data_dir, "test_ground_truth_private.csv")

df_train.to_csv(train_path, index=False)
df_test.to_csv(test_path, index=False)
df_test_full.to_csv(test_truth_path, index=False)

print(f"Successfully saved train.csv: {len(df_train)} rows -> {train_path}")
print(f"Successfully saved test.csv: {len(df_test)} rows -> {test_path}")
print("Train columns:", list(df_train.columns))
print("Test columns:", list(df_test.columns))
print("Train label distribution:\n", df_train["label"].value_counts())
