# src/truth_probe_sweep_layers.py

from truth_probe import train_truth_probe

# Try a few layer indices: last 4 layers, for example
LAYER_IDXS = [-1, -2, -3, -4]

if __name__ == "__main__":
    for layer_idx in LAYER_IDXS:
        print("\n==============================")
        print(f"Training probe for layer_idx={layer_idx}")
        print("==============================")
        v_truth, best_t, auc = train_truth_probe(layer_idx=layer_idx, max_examples=400)
