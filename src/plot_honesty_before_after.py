# src/plot_honesty_before_after.py
#
# Make a bar chart showing baseline vs honesty-filter metrics:
#   - Honest accuracy
#   - Dishonesty rate under Opposite-Day prompt
#
# Uses the numbers you observed with the improved prompt + tuned filter.

import os
import numpy as np
import matplotlib.pyplot as plt


def main():
    # === Plug in your measured metrics ===
    # From honesty_filter_tuned_improved_prompt.py:
    # Baseline (no filter):
    baseline_honest_acc = 0.825   # 165 / 200
    baseline_dishonesty = 0.364   # 60 / 165

    # Filtered setting (e.g. delta = 0.45):
    filtered_honest_acc = 0.810   # from your logs
    filtered_dishonesty = 0.182   # from your logs

    labels = ["Baseline", "Honesty filter (δ = 0.45)"]

    accs = [baseline_honest_acc, filtered_honest_acc]
    dishs = [baseline_dishonesty, filtered_dishonesty]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots()

    # Left bars: honest accuracy
    ax.bar(x - width / 2, accs, width, label="Honest accuracy")

    # Right bars: dishonesty rate
    ax.bar(x + width / 2, dishs, width, label="Dishonesty rate")

    ax.set_ylabel("Rate")
    ax.set_title("Honest accuracy vs dishonesty\nBefore vs after honesty filter")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=10)
    ax.set_ylim(0.0, 1.0)
    ax.legend()

    fig.tight_layout()

    os.makedirs("plots", exist_ok=True)
    out_path = os.path.join("plots", "honesty_before_after.png")
    fig.savefig(out_path, dpi=200)
    print(f"Saved bar chart to {out_path}")


if __name__ == "__main__":
    main()
