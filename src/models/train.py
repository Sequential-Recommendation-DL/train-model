from src.models.deepfm import DeepFM, CTRDataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os

def train_deepfm(epochs=10, batch_size=512, lr=0.001, save_path="models/deepfm_model.pth"):
    # Load dataset
    # train_data = CTRDataset("data/raw/ctr/train.csv")
    # train_data = CTRDataset("data/processed/train_clean.csv")
    train_data = CTRDataset("data/processed/train_split.csv")
    
    
    # test_data = CTRDataset("data/raw/ctr/test.csv",
    #                        feat_mapper=train_data.feat_mapper,
    #                        defaults=train_data.defaults)
    
    
    val_data = CTRDataset(
                        "data/processed/val_split.csv",
                        feat_mapper=train_data.feat_mapper,
                        defaults=train_data.defaults )   
    test_data = CTRDataset(
                        "data/processed/test_split.csv",
                        feat_mapper=train_data.feat_mapper,
                        defaults=train_data.defaults
                    )
    val_loader = DataLoader(val_data, batch_size=batch_size)

    # test_data = CTRDataset(
    #                     "data/processed/test_clean.csv",
    #                     feat_mapper=train_data.feat_mapper,
    #                                         defaults=train_data.defaults)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    # Init model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepFM(train_data.field_dims).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Train loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device).unsqueeze(1)
            preds = model(X)
            loss = criterion(preds, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Eval
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            # for X, y in test_loader:
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                preds = (model(X) > 0.5).float().squeeze()
                correct += (preds == y).sum().item()
                total += y.size(0)

        # print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}, Acc: {correct/total:.4f}")
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}, Avg Loss: {avg_loss:.4f}, Val Acc: {correct/total:.4f}")
    # Save model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print("Model saved to:", os.path.abspath(save_path))
    
    print("\n FINAL TEST")
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)

            preds = (model(X) > 0.5).float().squeeze()

            correct += (preds == y).sum().item()
            total += y.size(0)

    print(f"Test Acc: {correct/total:.4f}")

if __name__ == "__main__":
    train_deepfm()

#note file train này chạy lệnh  python -m src.models.train
# Python sẽ biên dịch file .py thành bytecode để chạy nhanh hơn. Những file biên dịch này được lưu trong thư mục __pycache__ với tên dạng:

# deepfm.cpython-313.pyc

# train.cpython-313.pyc
# Ý nghĩa
# .pyc là Python compiled file (bytecode).

# Nó giúp lần chạy sau nhanh hơn vì Python không cần dịch lại từ đầu.

# Nội dung bên trong là mã máy, không phải để bạn đọc hay chỉnh sửa.