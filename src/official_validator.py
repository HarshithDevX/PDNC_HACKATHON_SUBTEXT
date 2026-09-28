"""
Submission Validator & Evaluator for Problem 03: Context-Aware Sarcasm Detection
- Automatically checks submission format, column names, missing values, and valid labels.
- Validates and enforces exact ID row ordering matching the official test set.
- Automatically saves the verified & ordered file into the `submission/` folder.
- If run by organizers (with evaluation key), computes Macro F1 and records in results.csv.

Usage:
    # Student verification:
    python utils/validator.py
    python utils/validator.py --file submission/my_submission.csv

    # Organizer evaluation:
    python utils/validator.py --file submission/my_submission.csv --eval
"""

import os
import sys
import argparse
from datetime import datetime
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
SUBMISSION_DIR = os.path.join(BASE_DIR, "submission")
RESULTS_CSV = os.path.join(BASE_DIR, "results.csv")
EVAL_KEY_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "organizer_evaluation_keys"))
DEFAULT_EVAL_KEY = os.path.join(EVAL_KEY_DIR, "03_sarcasm_solution.csv")

VALID_LABELS = ['NOT_SARCASM', 'SARCASM']

def normalize_label(val):
    val_str = str(val).strip().upper()
    if val_str in ['1', 'SARCASM', 'SARCASTIC', 'TRUE', 'YES']:
        return 'SARCASM'
    elif val_str in ['0', 'NOT_SARCASM', 'NOT SARCASTIC', 'NON-SARCASM', 'FALSE', 'NO']:
        return 'NOT_SARCASM'
    return val_str

def find_candidate_file(explicit_file=None):
    if explicit_file:
        if os.path.exists(explicit_file):
            return explicit_file
        candidate = os.path.join(SUBMISSION_DIR, explicit_file)
        if os.path.exists(candidate):
            return candidate
        raise FileNotFoundError(f"Specified submission file not found: {explicit_file}")
    
    if os.path.exists(SUBMISSION_DIR):
        csv_files = [os.path.join(SUBMISSION_DIR, f) for f in os.listdir(SUBMISSION_DIR) if f.endswith('.csv') and f != 'submission_verified.csv']
        if csv_files:
            csv_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            return csv_files[0]

    sample = os.path.join(DATASET_DIR, "test", "sample_submission.csv")
    if os.path.exists(sample):
        print("[INFO] No file found in submission/ folder. Running verification using dataset/test/sample_submission.csv.")
        return sample
        
    raise FileNotFoundError("No candidate submission CSV found! Please place your file in submission/submission.csv")

def validate_submission(file_path, do_eval=False, solution_key=None):
    print("="*65)
    print("SARCASM SUBMISSION VALIDATOR & ORDER VERIFIER")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Target Submission: {file_path}")
    print("="*65)

    ref_test_path = os.path.join(DATASET_DIR, "test", "test.csv")
    if not os.path.exists(ref_test_path):
        raise FileNotFoundError(f"Reference test file missing: {ref_test_path}")

    ref_test_df = pd.read_csv(ref_test_path)
    expected_ids = ref_test_df['id'].tolist()
    expected_count = len(expected_ids)

    try:
        sub_df = pd.read_csv(file_path)
    except Exception as e:
        print(f"[ERROR] Failed to read CSV file: {e}")
        return False

    sub_df.columns = [c.strip().lower() for c in sub_df.columns]
    if 'id' not in sub_df.columns or 'prediction' not in sub_df.columns:
        print(f"[ERROR] Missing required columns! Expected ['id', 'prediction'], found {sub_df.columns.tolist()}")
        return False

    null_count = sub_df['prediction'].isnull().sum()
    if null_count > 0:
        print(f"[WARNING] Found {null_count} null/missing predictions! Filling with 'NOT_SARCASM'.")
        sub_df['prediction'] = sub_df['prediction'].fillna('NOT_SARCASM')

    submitted_ids = sub_df['id'].tolist()
    is_ordered = (submitted_ids == expected_ids)
    
    if not is_ordered:
        print("[NOTICE] Submission IDs are out of order or mismatched. Re-aligning to official test sequence...")
        sub_indexed = sub_df.drop_duplicates(subset=['id']).set_index('id')
        missing_ids = set(expected_ids) - set(sub_indexed.index)
        if missing_ids:
            print(f"[WARNING] Submission is missing {len(missing_ids)} expected IDs! Missing IDs will default to 'NOT_SARCASM'.")
            
        sub_ordered = sub_indexed.reindex(expected_ids, fill_value='NOT_SARCASM').reset_index()
        sub_ordered.rename(columns={'index': 'id'}, inplace=True)
        sub_df = sub_ordered
        order_corrected = True
        print("[SUCCESS] Successfully re-ordered all rows to match official test sequence.")
    else:
        order_corrected = False
        print("[SUCCESS] Submission row count and ID ordering match test set exactly.")

    sub_df['prediction'] = sub_df['prediction'].apply(normalize_label)

    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    verified_path = os.path.join(SUBMISSION_DIR, "submission_verified.csv")
    sub_df[['id', 'prediction']].to_csv(verified_path, index=False)
    print(f"[SAVED] Verified & ordered submission saved to: {verified_path}")
    print(f"[STATUS] Passed validation! Ready for evaluation ({len(sub_df)} rows).")

    key_path = solution_key if solution_key else DEFAULT_EVAL_KEY
    if do_eval or (key_path and os.path.exists(key_path)):
        if os.path.exists(key_path):
            print("\n" + "-"*40)
            print("RUNNING BENCHMARK EVALUATION (ORGANIZER MODE):")
            gt_df = pd.read_csv(key_path)
            merged = pd.merge(gt_df, sub_df, on='id', suffixes=('_gt', '_pred'))
            gt_col = 'label' if 'label' in gt_df.columns else ('label_gt' if 'label_gt' in merged.columns else 'label')
            if gt_col not in merged.columns and 'label_id' in merged.columns:
                gt_col = 'label_id'

            y_true = merged[gt_col].apply(normalize_label)
            y_pred = merged['prediction']

            acc = accuracy_score(y_true, y_pred)
            macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
            sarcasm_f1 = f1_score(y_true == 'SARCASM', y_pred == 'SARCASM', zero_division=0)
            cm = confusion_matrix(y_true, y_pred, labels=['NOT_SARCASM', 'SARCASM'])

            print(f"  Matched Rows:          {len(merged)} / {expected_count}")
            print(f"  Accuracy:              {acc * 100:.2f}%")
            print(f"  Macro F1 (Primary):    {macro_f1 * 100:.2f}%")
            print(f"  Sarcasm Class F1:      {sarcasm_f1 * 100:.2f}%")
            print("\nConfusion Matrix (Rows=True, Cols=Pred):")
            print(f"               Pred NOT_SARCASM  Pred SARCASM")
            print(f"True NOT_SARCASM:     {cm[0][0]:<15} {cm[0][1]:<15}")
            print(f"True SARCASM:         {cm[1][0]:<15} {cm[1][1]:<15}")
            print("-"*40)

            file_exists = os.path.exists(RESULTS_CSV)
            with open(RESULTS_CSV, "a", newline="", encoding="utf-8") as f:
                writer = pd.DataFrame([{
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'submission_file': os.path.basename(file_path),
                    'matched_samples': len(merged),
                    'total_samples': expected_count,
                    'order_corrected': order_corrected,
                    'accuracy': round(acc, 4),
                    'macro_f1': round(macro_f1, 4),
                    'sarcasm_f1': round(sarcasm_f1, 4),
                    'status': 'PASSED'
                }])
                writer.to_csv(f, header=not file_exists, index=False)
            print(f"[LOGGED] Evaluation score logged to: {RESULTS_CSV}")

    return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Validate & Order Sarcasm Submissions")
    parser.add_argument('--file', type=str, default=None, help='Path to submission CSV')
    parser.add_argument('--eval', action='store_true', help='Evaluate if solution key is available')
    parser.add_argument('--solution_key', type=str, default=None, help='Path to solution key CSV')
    args = parser.parse_args()

    try:
        target = find_candidate_file(args.file)
        validate_submission(target, do_eval=args.eval, solution_key=args.solution_key)
    except Exception as e:
        print(f"\n[FATAL ERROR] {e}")
        sys.exit(1)
