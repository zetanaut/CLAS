#!/usr/bin/env python3
"""Train a leakage-safe baseline classifier for H versus N toy events."""

from __future__ import annotations

import argparse

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from make_rgc_plots import read_events


FEATURES = (
    "x", "q2", "w", "y", "z", "phi_h", "p_t", "pion_p",
    "pion_theta", "mm2_e_pi", "mm_e_pi", "vz", "rxy",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input")
    parser.add_argument("--positive-target", type=int, choices=(1, 2, 3, 4), default=1)
    parser.add_argument("--negative-target", type=int, choices=(1, 2, 3, 4), default=2)
    args = parser.parse_args()
    data = read_events(args.input)
    mask = np.isin(data["target"], (args.positive_target, args.negative_target))
    x = np.column_stack([data[name][mask] for name in FEATURES])
    y = (data["target"][mask] == args.positive_target).astype(int)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.30, random_state=123, stratify=y
    )
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=2000),
    )
    model.fit(x_train, y_train)
    score = model.predict_proba(x_test)[:, 1]
    prediction = (score >= 0.5).astype(int)

    print(f"Events used: {len(y)} (positive={y.sum()}, negative={(y == 0).sum()})")
    print(f"ROC AUC: {roc_auc_score(y_test, score):.4f}")
    print(classification_report(y_test, prediction, target_names=(f"target-{args.negative_target}", f"target-{args.positive_target}"), digits=4))
    coefficients = model[-1].coef_[0]
    ranking = sorted(zip(FEATURES, coefficients), key=lambda item: abs(item[1]), reverse=True)
    print("Standardized logistic coefficients:")
    for name, coefficient in ranking:
        print(f"  {name:12s} {coefficient:+.4f}")
    print("The target label is used only as the training target, never as a feature.")


if __name__ == "__main__":
    main()
