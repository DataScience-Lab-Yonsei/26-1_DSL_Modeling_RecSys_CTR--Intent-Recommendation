import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, log_loss

from src.config import DEVICE


def log(msg: str) -> None:
    print(f"[INFO] {msg}", flush=True)


def compute_ece(y_true, y_pred, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(y_true)

    for i in range(n_bins):
        mask = (y_pred >= bins[i]) & (y_pred < bins[i + 1])
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / total) * abs(y_true[mask].mean() - y_pred[mask].mean())

    return ece


def train_model(model, train_loader, n_epochs=5, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=1e-5
    )
    criterion = nn.BCELoss()
    model.to(DEVICE)

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            label = batch["label"].to(DEVICE)
            sparse = batch["sparse"].to(DEVICE)
            dense = batch["dense"].to(DEVICE)

            u_emb = batch.get("user_emb")
            a_emb = batch.get("ad_emb")
            if u_emb is not None:
                u_emb = u_emb.to(DEVICE)
                a_emb = a_emb.to(DEVICE)

            pred = model(sparse, dense, u_emb, a_emb)
            loss = criterion(pred, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        current_lr = scheduler.get_last_lr()[0]
        log(f"Epoch {epoch + 1}/{n_epochs} - loss: {avg_loss:.4f}, lr: {current_lr:.6f}")
        scheduler.step()

    return model


def evaluate_model(model, loader, group_df=None):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            label = batch["label"]
            sparse = batch["sparse"].to(DEVICE)
            dense = batch["dense"].to(DEVICE)

            u_emb = batch.get("user_emb")
            a_emb = batch.get("ad_emb")
            if u_emb is not None:
                u_emb = u_emb.to(DEVICE)
                a_emb = a_emb.to(DEVICE)

            pred = model(sparse, dense, u_emb, a_emb).cpu().numpy()
            all_preds.append(pred)
            all_labels.append(label.numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)

    auc = roc_auc_score(y_true, y_pred)
    logloss = log_loss(y_true, y_pred)
    ece = compute_ece(y_true, y_pred)

    print(f"  AUC:     {auc:.4f}")
    print(f"  LogLoss: {logloss:.4f}")
    print(f"  ECE:     {ece:.4f}")

    if group_df is not None:
        groups = group_df["action_group"].values
        for g in ["sparse", "medium", "dense"]:
            mask = groups == g
            if mask.sum() < 100:
                continue
            g_auc = roc_auc_score(y_true[mask], y_pred[mask])
            print(f"  AUC [{g:6s}]: {g_auc:.4f} (n={mask.sum():,})")

    return {
        "auc": auc,
        "logloss": logloss,
        "ece": ece,
        "y_pred": y_pred,
        "y_true": y_true,
    }