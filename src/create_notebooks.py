import json
import os

notebooks_dir = r"C:\Users\donda\.gemini\antigravity-ide\scratch\SUBTEXT\notebooks"
os.makedirs(notebooks_dir, exist_ok=True)

def make_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.12.14"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

def markdown_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.strip().split("\n")]
    }

def code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.strip().split("\n")]
    }

# 1. 01_eda.ipynb
eda_nb = make_notebook([
    markdown_cell("# 01 - Exploratory Data Analysis (EDA)\n## SUBTEXT: Context-Aware Sarcasm & Irony Detection\n**Official Benchmark:** 4,200 train / 2,000 test | Primary Metric: Macro F1"),
    code_cell("""import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sys.path.append("../src")
import config
from data_loader import load_train_data, load_test_data, inspect_dataset

df_train = load_train_data()
df_test = load_test_data()

print("Train shape:", df_train.shape)
print("Test shape:", df_test.shape)
inspect_dataset(df_train, "Train")
"""),
    markdown_cell("### Class Distribution\nAnalyzing balance between SARCASM and NOT_SARCASM:"),
    code_cell("""print(df_train['label'].value_counts())
df_train['label'].value_counts().plot(kind='bar', color=['#2563eb', '#dc2626'])
plt.title("Training Label Distribution")
plt.ylabel("Count")
plt.show()
"""),
    markdown_cell("### Conversational Context Lengths\nChecking word count distributions of parent_context, full_dialogue_context, and response:"),
    code_cell("""df_train['resp_len'] = df_train['response'].apply(lambda x: len(str(x).split()))
df_train['parent_len'] = df_train['parent_context'].apply(lambda x: len(str(x).split()))
df_train['full_len'] = df_train['full_dialogue_context'].apply(lambda x: len(str(x).split()))

df_train[['resp_len', 'parent_len', 'full_len']].describe()
""")
])

# 2. 02_baseline.ipynb
baseline_nb = make_notebook([
    markdown_cell("# 02 - Baseline Experiments\n## TF-IDF + Logistic Regression Across Context Strategies\nEstablishes quantitative reference for technical defense."),
    code_cell("""import sys
import os
sys.path.append("../src")
from baseline import run_baseline_experiments

df_results = run_baseline_experiments()
df_results
""")
])

# 3. 03_training.ipynb
training_nb = make_notebook([
    markdown_cell("# 03 - Transformer Model Training\n## Context-Aware Cross-Encoder Fine-Tuning\nModel: DistilBERT Cross-Encoder | Strategy: FULL_CONTEXT_RESPONSE"),
    code_cell("""import sys
import os
sys.path.append("../src")
from train import train_model

# Run training pipeline with validation checkpointing
metadata = train_model(num_epochs=3, batch_size=16)
print("Best Validation Macro F1:", metadata["metrics"]["validation_macro_f1"])
""")
])

# 4. 04_evaluation.ipynb
eval_nb = make_notebook([
    markdown_cell("# 04 - Evaluation, Threshold Tuning & Test Inference\n## Comprehensive Metric Analysis, Error Diagnosis & Validator"),
    code_cell("""import sys
import os
sys.path.append("../src")
from inference import run_test_inference
from validator import validate_submission

# Generate official 2,000 predictions
sub_path = run_test_inference()

# Execute Organizer Validator
is_valid, errors = validate_submission(sub_path)
print("Validator Passed:", is_valid)
""")
])

notebooks = {
    "01_eda.ipynb": eda_nb,
    "02_baseline.ipynb": baseline_nb,
    "03_training.ipynb": training_nb,
    "04_evaluation.ipynb": eval_nb
}

for name, nb in notebooks.items():
    p = os.path.join(notebooks_dir, name)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"Created notebook: {p}")
