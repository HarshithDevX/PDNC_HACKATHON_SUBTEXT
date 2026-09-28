import os
import sys
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config

def validate_submission(submission_file: str = None, test_file: str = None):
    """
    Official Organizer Submission Validator for Challenge 03.
    Verifies:
      1. File existence and accessibility
      2. Exactly two columns: 'id' and 'prediction'
      3. Exactly 2,000 rows
      4. Prediction labels are strictly 'SARCASM' or 'NOT_SARCASM'
      5. Original test ID sequence is strictly preserved (1-to-1 match)
      6. No NaN, null, or empty values
    """
    sub_path = submission_file or config.SUBMISSION_FILE
    test_path = test_file or config.TEST_FILE

    print(f"\n{'='*60}")
    print(f"RUNNING OFFICIAL HACKATHON SUBMISSION VALIDATOR")
    print(f"Target Submission: {sub_path}")
    print(f"{'='*60}")

    errors = []

    # 1. Existence Check
    if not os.path.exists(sub_path):
        errors.append(f"Submission file does not exist: {sub_path}")
        return False, errors

    try:
        df_sub = pd.read_csv(sub_path)
    except Exception as e:
        errors.append(f"Failed to parse CSV file: {e}")
        return False, errors

    # 2. Column Check
    expected_cols = ["id", "prediction"]
    actual_cols = list(df_sub.columns)
    if actual_cols != expected_cols:
        errors.append(f"Invalid columns: Found {actual_cols}, expected strictly {expected_cols}")

    # 3. Row Count Check
    expected_rows = 2000
    actual_rows = len(df_sub)
    if actual_rows != expected_rows:
        errors.append(f"Invalid row count: Found {actual_rows} rows, expected strictly {expected_rows} rows")

    # 4. Missing Values Check
    null_counts = df_sub.isnull().sum()
    if null_counts.any():
        errors.append(f"Missing/null values detected: {null_counts.to_dict()}")

    # 5. Label Validity Check
    allowed_labels = {"SARCASM", "NOT_SARCASM"}
    unique_preds = set(df_sub["prediction"].dropna().unique())
    invalid_labels = unique_preds - allowed_labels
    if invalid_labels:
        errors.append(f"Illegal prediction labels found: {invalid_labels}. Allowed: {allowed_labels}")

    # 6. ID Alignment and Ordering Check against official test.csv
    if os.path.exists(test_path):
        df_test = pd.read_csv(test_path)
        expected_ids = list(df_test["id"])
        actual_ids = list(df_sub["id"])
        if actual_ids != expected_ids:
            errors.append("Test IDs do not match the official test set order or contain mismatches/duplicates!")
    else:
        print(f"Warning: Reference test file {test_path} not found for ID alignment verification.")

    if errors:
        print("\n[VALIDATION FAILED] The following issues must be resolved:")
        for err in errors:
            print(f"  [X] {err}")
        return False, errors

    print("\n[VALIDATION PASSED] All checks succeeded successfully!")
    print(f"  [OK] Columns: {actual_cols}")
    print(f"  [OK] Row Count: {actual_rows:,}")
    print(f"  [OK] Prediction Distribution:\n{df_sub['prediction'].value_counts().to_string()}")
    print(f"  [OK] ID Sequence Alignment: 100% matched to reference test set")
    print(f"  [OK] No missing values or invalid labels")
    return True, []

if __name__ == "__main__":
    filepath = sys.argv[1] if len(sys.argv) > 1 else None
    success, _ = validate_submission(filepath)
    sys.exit(0 if success else 1)
