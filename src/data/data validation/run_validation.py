from src.data.validate import DataValidator

validator = DataValidator()

# Validate train set
validator.validate_file(
    input_path="data/raw/ctr/train.csv",
    output_path="data/processed/train_clean.csv"
)

# Validate test set
validator.validate_file(
    input_path="data/raw/ctr/test.csv",
    output_path="data/processed/test_clean.csv"
)