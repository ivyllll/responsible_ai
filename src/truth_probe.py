# src/truth_probe.py
#
# Learn a "truth direction" from Mistral hidden states on the True–False dataset.
# - Uses 400 examples (fixed random sample) from true-false-dataset
# - Trains a logistic regression probe on layer_idx
# - Orients the direction so higher score = "more likely TRUE"
# - Tunes a threshold on train scores
# - Reports test accuracy + AUC
# - Saves a ROC curve image and the truth direction vector

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from collect_activations import collect_hidden_states_for_true_false

# ---------------------------------------------------------------------
# Helper: search for best threshold on a continuous score
# ---------------------------------------------------------------------

def find_best_threshold(scores, y_true):
    """
    Given continuous scores and true labels (0/1),
    search over a small set of thresholds to maximize accuracy.
    """
    candidates = [0.0]
    qs = np.linspace(0.1, 0.9, 9)
    candidates += list(np.quantile(scores, qs))

    best_acc = 0.0
    best_t = 0.0
    for t in candidates:
        preds = (scores > t).astype(int)
        acc = accuracy_score(y_true, preds)
        if acc > best_acc:
            best_acc = acc
            best_t = t
    return best_t, best_acc


# ---------------------------------------------------------------------
# Main probe training function
# ---------------------------------------------------------------------

def train_truth_probe(
    layer_idx: int = -4,
    max_examples: int = 400,
    save_prefix: str | None = None,
):
    """
    Train a linear probe (logistic regression) to predict true vs false
    from hidden states at `layer_idx`.

    Steps:
      1. Collect hidden states X and labels y from the True–False dataset.
      2. Train/test split.
      3. Train LogisticRegression (linear classifier).
      4. Ensure orientation so that higher scores = "more likely TRUE".
      5. Tune a decision threshold on train scores.
      6. Report test accuracy and AUC.
      7. Save a ROC curve image for this layer.

    Returns:
      v_truth : np.ndarray, shape (hidden_dim,)
          Oriented truth direction vector.
      best_t : float
          Best threshold on that direction for classification.
      auc    : float
          AUC on the held-out test set (with oriented scores).
      acc_test : float
          Final accuracy on the held-out test set (with tuned threshold).
    """
    # 1. Collect activations and labels
    X, y = collect_hidden_states_for_true_false(
        layer_idx=layer_idx,
        max_examples=max_examples,
    )

    # 2. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Fit logistic regression probe
    clf = LogisticRegression(max_iter=2000)
    clf.fit(X_train, y_train)

    # Scores before orientation (w·x + b)
    scores_train_raw = clf.decision_function(X_train)
    scores_test_raw = clf.decision_function(X_test)
    auc_raw = roc_auc_score(y_test, scores_test_raw)
    print(f"[BEFORE ORIENTATION] AUC on test: {auc_raw:.3f}")

    v_truth = clf.coef_[0].copy()

    # 4. Orientation: make higher scores mean "more likely TRUE"
    if auc_raw < 0.5:
        print("AUC < 0.5 → flipping direction to align with TRUE.")
        v_truth = -v_truth
        scores_train = -scores_train_raw
        scores_test = -scores_test_raw
        auc = roc_auc_score(y_test, scores_test)
        print(f"[AFTER ORIENTATION] AUC on test: {auc:.3f}")
    else:
        print("AUC ≥ 0.5 → keeping original orientation as TRUE direction.")
        scores_train = scores_train_raw
        scores_test = scores_test_raw
        auc = auc_raw

    # 5. Threshold tuning on train scores
    best_t, best_acc_train = find_best_threshold(scores_train, y_train)
    print(f"Best threshold on train: t={best_t:.4f}, train acc={best_acc_train:.3f}")

    preds_test = (scores_test > best_t).astype(int)
    acc_test = accuracy_score(y_test, preds_test)
    print(f"Final test accuracy with tuned threshold: {acc_test:.3f}, AUC: {auc:.3f}")

    print("Final truth direction shape:", v_truth.shape)

    # 6. ROC curve plot (for presentation)
    fpr, tpr, _ = roc_curve(y_test, scores_test)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random (AUC = 0.5)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - Layer {layer_idx}")
    plt.legend(loc="lower right")

    # Decide filename
    if save_prefix is None:
        out_path = f"roc_layer{layer_idx}_n{max_examples}.png"
    else:
        out_path = f"{save_prefix}_roc_layer{layer_idx}_n{max_examples}.png"

    plt.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close()
    print(f"Saved ROC curve to {out_path}")

    return v_truth, best_t, auc, acc_test


# ---------------------------------------------------------------------
# Script entry point: this is your "final" truth direction
# ---------------------------------------------------------------------

if __name__ == "__main__":
    LAYER_IDX = -4
    MAX_EXAMPLES = 400

    v_truth, best_t, auc, acc_test = train_truth_probe(
        layer_idx=LAYER_IDX,
        max_examples=MAX_EXAMPLES,
        save_prefix="final",
    )

    np.save(f"v_truth_layer{LAYER_IDX}_n{MAX_EXAMPLES}.npy", v_truth)
    print(f"Saved oriented v_truth to v_truth_layer{LAYER_IDX}_n{MAX_EXAMPLES}.npy")
    print("You can use best_t =", best_t, "as the threshold for classification if needed.")
    print(f"(Summary) Layer {LAYER_IDX}, N={MAX_EXAMPLES}: AUC={auc:.3f}, acc={acc_test:.3f}")
