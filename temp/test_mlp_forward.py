#src/test_mlp_forward.py

# test_mlp_forward.py

import torch

from src.test_model import MLPBaseline
from src.dataloader import create_dataloaders




def main():
    # -----------------------------
    # 1. Device setting
    # -----------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # -----------------------------
    # 2. Dataloader
    # -----------------------------
    data_path = "data/processed/final_ms_mca_dataset_2024_01.csv"

    feature_cols = [
        "no2_mean",
        "nightlight_mean",
    ]

    train_loader, val_loader, test_loader, _ = create_dataloaders(
    csv_path=data_path,
    feature_cols=feature_cols,
    target_cls_col="label_cls",
    target_reg_col="label_reg_log",
    batch_size=32,
    test_size=0.2,
    val_size=0.2,
    random_state=42,
)

    # -----------------------------
    # 3. Model
    # -----------------------------
    model = MLPBaseline(
        input_dim=len(feature_cols),
        hidden_dim=64,
        num_classes=5,
        dropout=0.2,
    ).to(device)

    model.eval()

    # -----------------------------
    # 4. Forward check
    # -----------------------------
    for batch in train_loader:
        x, y_cls, y_reg = batch

        x = x.to(device)
        y_cls = y_cls.to(device)
        y_reg = y_reg.to(device)

        with torch.no_grad():
            logits = model(x)

        preds = torch.argmax(logits, dim=1)
        probs = torch.softmax(logits, dim=1)

        print("===== MLP Forward Check =====")
        print("x shape:", x.shape)
        print("logits shape:", logits.shape)
        print("probs shape:", probs.shape)
        print("preds shape:", preds.shape)
        print("y_cls shape:", y_cls.shape)

        print("\nlogits example:")
        print(logits[0])

        print("\nprobability example:")
        print(probs[0])

        print("\npred example:", preds[0])
        print("true example:", y_cls[0])

        break


if __name__ == "__main__":
    main()



