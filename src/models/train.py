
# import os

# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader


# def train(
#     model,
#     train_loader: DataLoader,
#     val_loader: DataLoader,
#     epochs: int = 10,
#     lr: float = 1e-3,
#     device: str = "cpu",
#     save_path: str = "models/deepfm_best.pth",
#     patience: int = 3,         # early stopping: dừng sau N epoch val_loss không giảm
# ):
#     model = model.to(device)

#     # BCEWithLogitsLoss ổn định hơn BCELoss + sigmoid riêng
#     criterion = nn.BCEWithLogitsLoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

#     history = []
#     best_val_loss = float("inf")
#     best_val_acc = 0.0
#     patience_counter = 0

#     for epoch in range(epochs):
#         # --- Huấn luyện ---
#         model.train()
#         total_loss = 0.0
#         for X, y in train_loader:
#             X, y = X.to(device), y.to(device).unsqueeze(1)
#             preds = model(X)
#             loss = criterion(preds, y)

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()

#         train_loss = total_loss / len(train_loader)

#         # --- Kiểm tra trên tập validation ---
#         model.eval()
#         correct, total = 0, 0
#         val_total_loss = 0.0
#         with torch.no_grad():
#             for X, y in val_loader:
#                 X, y = X.to(device), y.to(device)
#                 logits = model(X).squeeze()
#                 val_loss_batch = criterion(logits, y)
#                 val_total_loss += val_loss_batch.item()

#                 preds = (torch.sigmoid(logits) > 0.5).float()
#                 correct += (preds == y).sum().item()
#                 total += y.size(0)

#         val_loss = val_total_loss / len(val_loader)
#         val_acc = correct / total

#         status = ""

#         # early stopping
#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             best_val_acc = val_acc
#             patience_counter = 0
#             os.makedirs(os.path.dirname(save_path), exist_ok=True)
#             torch.save(model.state_dict(), save_path)
#             status += f"  → Đã lưu model tốt nhất (val_loss={best_val_loss:.4f})"
#         else:
#             patience_counter += 1
#             status += f"  ⚠ Val loss không giảm ({patience_counter}/{patience})"

#         print(f"Epoch {epoch+1}/{epochs}  Train Loss: {train_loss:.4f}  Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.4f}{status}")

#         history.append({
#             "epoch": epoch + 1,
#             "train_loss": train_loss,
#             "val_loss": val_loss,
#             "val_acc": val_acc,
#         })

#         if patience_counter >= patience:
#             print(f"\n  Dừng sớm tại epoch {epoch+1} — val loss không cải thiện sau {patience} epochs.")
#             break

#     print(f"\nHuấn luyện hoàn tất. Val Loss tốt nhất: {best_val_loss:.4f}  Val Acc tốt nhất: {best_val_acc:.4f}")
#     return model, history


# if __name__ == "__main__":
#     print("Hãy chạy pipeline.py thay vì file này: python -m src.pipeline.pipeline")
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def train(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 10,
    lr: float = 1e-3,
    device: str = "cpu",
    save_path: str = "models/deepfm_best.pth",
    patience: int = 3,
):
    model = model.to(device)

    # tính pos_weight để cân bằng imbalanced dataset
   
    # pos_weight = torch.tensor([num_neg / num_pos], device=device)
    # print(f"  Tỉ lệ: Thích={num_pos:,.0f} | Không thích={num_neg:,.0f} | pos_weight={pos_weight.item():.4f}")
    criterion = nn.BCEWithLogitsLoss()

    # BCEWithLogitsLoss với pos_weight cân bằng class
    # criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    history = []
    best_val_loss = float("inf")
    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(epochs):
        # --- Huấn luyện ---
        model.train()
        total_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device).unsqueeze(1)
            preds = model(X)
            loss = criterion(preds, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)

        # --- Kiểm tra trên tập validation ---
        model.eval()
        correct, total = 0, 0
        val_total_loss = 0.0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                logits = model(X).squeeze()
                val_loss_batch = criterion(logits, y)
                val_total_loss += val_loss_batch.item()

                preds = (torch.sigmoid(logits) > 0.5).float()
                correct += (preds == y).sum().item()
                total += y.size(0)

        val_loss = val_total_loss / len(val_loader)
        val_acc = correct / total
        status = ""

        # early stopping theo val_loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            patience_counter = 0
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            status += f"  → Đã lưu model tốt nhất (val_loss={best_val_loss:.4f})"
        else:
            patience_counter += 1
            status += f"  ⚠ Val loss không giảm ({patience_counter}/{patience})"

        print(f"Epoch {epoch+1}/{epochs}  Train Loss: {train_loss:.4f}  Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.4f}{status}")

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
        })

        if patience_counter >= patience:
            print(f"\n  Dừng sớm tại epoch {epoch+1} — val loss không cải thiện sau {patience} epochs.")
            break

    print(f"\nHuấn luyện hoàn tất. Val Loss tốt nhất: {best_val_loss:.4f}  Val Acc tốt nhất: {best_val_acc:.4f}")
    return model, history


if __name__ == "__main__":
    print("Hãy chạy pipeline.py thay vì file này: python -m src.pipeline.pipeline")