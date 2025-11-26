# src/truth_probe_roc.py
#
# Train a truth probe on Mistral hidden states (layer -4, N=400)
# and save a ROC curve plot for use in the report/presentation.
#
# Run:
#   cd /home/rxs1540
#   source .venv/bin/activate
#   python src/truth_probe_roc.py

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split

from model_utils import load_mistral
from collect_activations import collect_hidden_states_for_true_false
from data_loading import load_true_false_dataset


def main():
    print("Loading Mistral model...")
    model, tokenizer = load_mistral()
    model.eval()

    # 1. Collect hidden states for 400 examples at layer -4
    print("Collecting activations for ROC probe (layer -4, N=400)...")
    X, y = collect_hidden_states_for_true_false(
        layer_idx=-4,
        max_examples=400,
    )
    print(f"X shape: {X.shape}, y mean: {y.mean():.3f}")

    # 2. Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123, stratify=y
    )

    # 3. Train logistic regression probe
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    train_acc = clf.score(X_train, y_train)
    test_acc = clf.score(X_test, y_test)
    print(f"Probe train accuracy: {train_acc:.3f}")
    print(f"Probe test accuracy:  {test_acc:.3f}")

    # 4. ROC & AUC
    scores_test = clf.decision_function(X_test)
    fpr, tpr, _ = roc_curve(y_test, scores_test)
    auc = roc_auc_score(y_test, scores_test)
    print(f"Test AUC: {auc:.3f}")

    # 5. Plot ROC curve
    os.makedirs("plots", exist_ok=True)
    out_path = os.path.join("plots", "roc_truth_layer-4_n400.png")

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"Truth probe (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve – Truth Probe (Mistral, layer -4, N=400)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f"Saved ROC curve to {out_path}")


if __name__ == "__main__":
    main()
