import random
import os

def split_dataset(
    input_path,
    train_path,
    val_path,
    test_path,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    seed=42
):

    random.seed(seed)

    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    random.shuffle(lines)

    total = len(lines)

    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_lines = lines[:train_end]
    val_lines = lines[train_end:val_end]
    test_lines = lines[val_end:]

    os.makedirs(os.path.dirname(train_path), exist_ok=True)

    with open(train_path, "w", encoding="utf-8") as f:
        f.writelines(train_lines)

    with open(val_path, "w", encoding="utf-8") as f:
        f.writelines(val_lines)

    with open(test_path, "w", encoding="utf-8") as f:
        f.writelines(test_lines)

    print("DATA SPLIT")
    print("Total:", total)
    print("Train:", len(train_lines))
    print("Validation:", len(val_lines))
    print("Test:", len(test_lines))