import os

# Project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data directories and files
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OFFICIAL_DATA_DIR = os.path.join(DATA_DIR, "official")
TRAIN_FILE = os.path.join(OFFICIAL_DATA_DIR, "train.csv")
TEST_FILE = os.path.join(OFFICIAL_DATA_DIR, "test.csv")
SAMPLE_SUBMISSION_FILE = os.path.join(OFFICIAL_DATA_DIR, "sample_submission.csv")
ORGANIZER_VALIDATOR = os.path.join(OFFICIAL_DATA_DIR, "validator.py")

# Models and outputs directories
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
BEST_MODEL_DIR = os.path.join(MODELS_DIR, "best_model")
CHECKPOINT_EXP_A = os.path.join(MODELS_DIR, "checkpoint_exp_a")
CHECKPOINT_EXP_B = os.path.join(MODELS_DIR, "checkpoint_exp_b")
CHECKPOINT_EXP_C = os.path.join(MODELS_DIR, "checkpoint_exp_c")

OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")

METRICS_FILE = os.path.join(OUTPUTS_DIR, "metrics.json")
VAL_PRED_FILE = os.path.join(OUTPUTS_DIR, "validation_predictions.csv")
TEST_PRED_FILE = os.path.join(OUTPUTS_DIR, "test_predictions.csv")
SUBMISSION_FILE = os.path.join(OUTPUTS_DIR, "submission.csv")
EXPERIMENT_RESULTS_FILE = os.path.join(REPORTS_DIR, "experiment_results.csv")
EDA_REPORT_FILE = os.path.join(REPORTS_DIR, "eda_report.md")
ERROR_ANALYSIS_FILE = os.path.join(REPORTS_DIR, "error_analysis.md")

# Ensure required directories exist
for d in [DATA_DIR, MODELS_DIR, BEST_MODEL_DIR, CHECKPOINT_EXP_A, CHECKPOINT_EXP_B, CHECKPOINT_EXP_C, OUTPUTS_DIR, REPORTS_DIR]:
    os.makedirs(d, exist_ok=True)

# Training & Modeling Configuration
RANDOM_SEED = 42
VAL_SPLIT_SIZE = 0.2
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
MAX_SEQ_LENGTH = 128

# Backbone model architecture (Strictly DeBERTa-v3-base only)
MODEL_NAME = "microsoft/deberta-v3-base"

# Context Strategies:
# 1. 'RESPONSE_ONLY' (Experiment A)
# 2. 'PARENT_RESPONSE' (Experiment B)
# 3. 'FULL_CONTEXT_RESPONSE' (Experiment C)
DEFAULT_CONTEXT_STRATEGY = "FULL_CONTEXT_RESPONSE"

# Label mapping
LABEL2ID = {"NOT_SARCASM": 0, "SARCASM": 1}
ID2LABEL = {0: "NOT_SARCASM", 1: "SARCASM"}
