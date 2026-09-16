#!/usr/bin/env python3
"""Train a small PyTorch MLP to classify D versus N SIDIS events."""

from __future__ import annotations

import argparse

import numpy as np
import torch
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from make_rgc_plots import read_events


FEATURES = (
    "x", "q2", "w", "y", "z", "phi_h", "p_t", "pion_p",
    "pion_theta", "mm2_e_pi", "mm_e_pi", "vz", "rxy",
)


class MLP(nn.Module):
    def __init__(self, n_features: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_features, 64), nn.ReLU(), nn.Dropout(0.10),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input")
    parser.add_argument("--positive-target", type=int, choices=(1, 2, 3, 4), default=4)
    parser.add_argument("--negative-target", type=int, choices=(1, 2, 3, 4), default=2)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    data = read_events(args.input)
    mask = np.isin(data["target"], (args.positive_target, args.negative_target))
    x = np.column_stack([data[name][mask] for name in FEATURES]).astype(np.float32)
    y = (data["target"][mask] == args.positive_target).astype(np.float32)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.30, random_state=args.seed, stratify=y
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.20, random_state=args.seed, stratify=y_train
    )
    scaler = StandardScaler().fit(x_train)
    x_train, x_val, x_test = (scaler.transform(a).astype(np.float32) for a in (x_train, x_val, x_test))

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=args.batch_size, shuffle=True,
    )
    model = MLP(x_train.shape[1])
    positive_weight = torch.tensor([(y_train == 0).sum() / max((y_train == 1).sum(), 1)], dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)

    best_state, best_val, patience = None, float("inf"), 0
    x_val_t, y_val_t = torch.from_numpy(x_val), torch.from_numpy(y_val)
    for _epoch in range(args.epochs):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(criterion(model(x_val_t), y_val_t))
        if val_loss < best_val - 1e-4:
            best_val, patience = val_loss, 0
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        else:
            patience += 1
            if patience >= 20:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        score = torch.sigmoid(model(torch.from_numpy(x_test))).numpy()
    prediction = (score >= 0.5).astype(int)
    print(f"Events used: {len(y)} (positive={int(y.sum())}, negative={int((y == 0).sum())})")
    print(f"PyTorch MLP: {len(FEATURES)} -> 64 -> 32 -> 1, early-stopped validation loss={best_val:.4f}")
    print(f"ROC AUC: {roc_auc_score(y_test, score):.4f}")
    print(classification_report(y_test, prediction,
                                target_names=(f"target-{args.negative_target}", f"target-{args.positive_target}"),
                                digits=4))
    print("The target label is used only as the training target, never as a feature.")


if __name__ == "__main__":
    main()
