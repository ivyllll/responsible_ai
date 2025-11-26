# src/collect_activations.py

import numpy as np
import torch
from typing import Optional, Tuple

from data_loading import load_true_false_dataset
from model_utils import load_mistral

# LAT-style prompt to make the model "think about" truthfulness.
LAT_PROMPT = """You are analyzing the truthfulness of statements.

Consider the following statement:
"{statement}"

The truthfulness of this statement is:"""


def collect_hidden_states_for_true_false(
    layer_idx: int = -2,          # -1 = last layer, -2 = second to last
    max_examples: Optional[int] = 500,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    For each statement in the True-False dataset, run Mistral with LAT_PROMPT
    and collect the hidden state of the LAST token at `layer_idx`.

    Returns:
      X: [num_examples, hidden_dim] numpy array of activations
      y: [num_examples] numpy array of labels (0/1)
    """
    df = load_true_false_dataset()

    if max_examples is not None:
        df = df.sample(n=max_examples, random_state=42).reset_index(drop=True)

    print(f"Collecting activations for {len(df)} examples...")

    model, tokenizer = load_mistral()
    model.eval()

    reps = []
    labels = []

    for i, row in df.iterrows():
        statement = row["statement"]
        label = int(row["label"])

        # Build LAT-style prompt
        prompt = LAT_PROMPT.format(statement=statement)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            # FORWARD PASS, not generate()
            outputs = model(**inputs)  # returns hidden_states because config has output_hidden_states=True

        hidden_states = outputs.hidden_states  # tuple: (layer0,...,layerN)
        layer = hidden_states[layer_idx]       # shape: (batch, seq_len, dim)

        # Take the LAST token representation and cast to float32 so numpy can handle it
        last_token_vec = (
            layer[0, -1, :]
            .detach()
            .to(torch.float32)    # <-- important: bfloat16 -> float32
            .cpu()
            .numpy()
        )

        reps.append(last_token_vec)
        labels.append(label)

        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1} examples...")

    X = np.vstack(reps)      # [num_examples, hidden_dim]
    y = np.array(labels)     # [num_examples]

    print("Done. X shape:", X.shape, " y shape:", y.shape)
    return X, y


if __name__ == "__main__":
    # Small test run
    X, y = collect_hidden_states_for_true_false(layer_idx=-2, max_examples=200)
    print("X shape:", X.shape)
    print("y mean (should be around 0.5 if dataset is balanced):", y.mean())
