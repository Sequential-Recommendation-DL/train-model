from src.data.split_data import split_dataset

split_dataset(
    input_path="data/processed/train_clean.csv",

    train_path="data/processed/train_split.csv",
    val_path="data/processed/val_split.csv",
    test_path="data/processed/test_split.csv"
)